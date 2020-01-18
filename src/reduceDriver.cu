#ifndef _REDUCE_DRIVER_
#define _REDUCE_DRIVER_

#include "util.cu"
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

enum class ReduceOp
{
    DOT = 0,
    NORM2 = 1,
};


inline std::string OpToString(const ReduceOp ops) {
    return (ops == ReduceOp::DOT) ? "DOT" : "NORM2";
}

/**
* cublasReduceStream()
*/
inline float cublasReduceStream(const int N, double* d_r, double* d_p,
    const ReduceOp ops, const int num_ops, const int num_nodes, const double gold) {

    if (num_ops % num_nodes != 0) {
        fprintf(stderr, "cublasReduceStream():: num_ops should be divisible by num_nodes");
        exit(EXIT_FAILURE);
    }
    int num_runs = num_ops / num_nodes;

    double* d_res;
    CUDA_ERROR(cudaMalloc((void**)&d_res, sizeof(double)));
    CUDA_ERROR(cudaMemset((void*)d_res, 0, sizeof(double)));

    cudaStream_t cublas_stream;
    CUDA_ERROR(cudaStreamCreate(&cublas_stream));
    cublasHandle_t cublas_handle = 0;

    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_ERROR(cublasSetStream(cublas_handle, cublas_stream));

    cudaEvent_t start, end;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&end));
    CUDA_ERROR(cudaEventRecord(start, cublas_stream));
    for (int iter = 0; iter < num_ops; ++iter) {
        switch (ops)
        {
        case ReduceOp::DOT: {
            CUBLAS_ERROR(cublasDdot(cublas_handle, N, d_p, 1, d_r, 1, d_res));
            break;
        }
        case ReduceOp::NORM2: {
            CUBLAS_ERROR(cublasDnrm2(cublas_handle, N, d_r, 1, d_res));
            break;
        }
        default:
            break;
        }
        CUDA_ERROR(cudaStreamSynchronize(cublas_stream));
    }

    CUDA_ERROR(cudaEventRecord(end, cublas_stream));
    CUDA_ERROR(cudaEventSynchronize(end));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaStreamDestroy(cublas_stream));
    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, end));

    double h_res(0);
    CUDA_ERROR(cudaMemcpy((void*)&h_res, (void*)d_res, sizeof(double), cudaMemcpyDeviceToHost));

    if (std::abs(h_res - gold) > 0.00001) {
        fprintf(stderr, "cublasReduceGraph():: failed with N= %d", N);
        exit(EXIT_FAILURE);
    }

    CUDA_ERROR(cudaFree(d_res));


    return time / num_ops;
}


/**
* cublasReduceGraph()
*/
inline float cublasReduceGraph(const int N, double*d_r, double*d_p, const ReduceOp ops,
    const int num_ops, const int num_nodes, const double gold) {
    
    if (num_ops % num_nodes != 0) {
        fprintf(stderr, "cublasReduceGraph():: num_ops should be divisible by num_nodes");
        exit(EXIT_FAILURE);
    }
    int num_runs = num_ops / num_nodes;

    double* d_res;
    CUDA_ERROR(cudaMalloc((void**)&d_res, sizeof(double)));
    CUDA_ERROR(cudaMemset((void*)d_res, 0, sizeof(double)));

    cudaGraph_t cuda_graph;
    cudaStream_t capture_stream;
    cublasHandle_t cublas_handle = 0;
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CUDA_ERROR(cudaStreamCreate(&capture_stream));
    CUDA_ERROR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    CUBLAS_ERROR(cublasSetStream(cublas_handle, capture_stream));
    CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    for (int n = 0; n < num_nodes; ++n) {
        switch (ops)
        {
        case ReduceOp::DOT: {
            CUBLAS_ERROR(cublasDdot(cublas_handle, N, d_p, 1, d_r, 1, d_res));            
            break;
        }
        case ReduceOp::NORM2: {
            CUBLAS_ERROR(cublasDnrm2(cublas_handle, N, d_r, 1, d_res));           
            break;
        }
        default:
            break;
        }        
    }
    CUDA_ERROR(cudaStreamEndCapture(capture_stream, &cuda_graph));

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(cuda_graph, nodes, &generated_num_nodes));

    //This factor because different reduction ops (dot vs. norm) in CUBLAS produces
    //different number of nodes 
    int factor = (ops == ReduceOp::DOT) ? 2 : 4;

    if (generated_num_nodes != num_nodes* factor) {
        fprintf(stderr, "cublasReduceGraph():: CUDA Graph has generated %d but the expected is %d",
            static_cast<int>(generated_num_nodes), num_nodes);
        exit(EXIT_FAILURE);
    }
    cudaGraphExec_t cuda_graph_exec;
    CUDA_ERROR(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, capture_stream));
    for (int iter = 0; iter < num_runs; ++iter) {
        CUDA_ERROR(cudaGraphLaunch(cuda_graph_exec, capture_stream));
        CUDA_ERROR(cudaStreamSynchronize(capture_stream));
    }

    CUDA_ERROR(cudaEventRecord(stop, capture_stream));
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    CUDA_ERROR(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_ERROR(cudaGraphDestroy(cuda_graph));
    CUDA_ERROR(cudaStreamDestroy(capture_stream));
    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

    double h_res(0);
    CUDA_ERROR(cudaMemcpy((void*)&h_res, (void*)d_res, sizeof(double), cudaMemcpyDeviceToHost));
    
    if (std::abs(h_res - gold) > 0.00001) {
        fprintf(stderr, "cublasReduceGraph():: failed with N= %d", N);
        exit(EXIT_FAILURE);
    }
    
    CUDA_ERROR(cudaFree(d_res));
    return time / num_ops;

}

inline void reduceDriver(const int num_ops, const int num_nodes, 
    const int start, const int end,  const double max_bandwidth, const ReduceOp ops) {
    //
    CUDA_ERROR(cudaProfilerStart());
    std::cout << " ****** Reduce Driver ("<< OpToString (ops) <<") with " << num_ops << " operations and "
        << num_nodes << " nodes Started ******" << std::endl;
    const char separator = ' ';
    const int numWidth = 20;
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Exp (2^x)";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Size";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Theoretical Time";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Practical Time";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "CUBLAS GraphTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "CUBLAS StreamTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "HANDMADE GraphTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "HANDMADE StreamTime";
    std::cout << std::endl << std::endl;

    for (int exp = start; exp <= end; ++exp) {
        int N = 1 << exp;
        const double theoretical_time = 0;//TODO
        const double practical_time = 0;//TODO
        double* d_r(NULL), * d_p(NULL);

        CUDA_ERROR(cudaMalloc((void**)&d_r, N * sizeof(double)));
        if (ops == ReduceOp::DOT) {
            CUDA_ERROR(cudaMalloc((void**)&d_p, N * sizeof(double)));
        }
        std::vector<double> h_r(N, 0);
        std::vector<double>h_p;
        if (ops == ReduceOp::DOT) {
            h_p.resize(N, 0);
        }

        std::generate(h_r.begin(), h_r.end(), []() {return double(rand()) / double(RAND_MAX); });
        if (ops == ReduceOp::DOT) {
            std::generate(h_p.begin(), h_p.end(), []() {return double(rand()) / double(RAND_MAX); });
        }

        double gold(0);
        for (int i = 0; i < N; ++i) {
            switch (ops)
            {
            case ReduceOp::NORM2: {
                gold += h_r[i] * h_r[i];
                break;
            }
            case ReduceOp::DOT: {
                gold += h_p[i] * h_r[i];
                break;
            }            
            default:
                break;
            }            
        }
        if (ops == ReduceOp::NORM2) {
            gold = std::sqrt(gold);
        }

        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        if (ops == ReduceOp::DOT) {
            CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        }

        float cublas_graph_time = cublasReduceGraph(N, d_r, d_p, ops, num_ops,  num_nodes, gold);
        float cublas_stream_time = cublasReduceStream(N, d_r, d_p, ops, num_ops, num_nodes, gold);
        float handmade_graph_time = 0;
        float handmade_stream_time = 0;

        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << exp;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << N;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << theoretical_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << practical_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_graph_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_stream_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << handmade_graph_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << handmade_stream_time;
        //std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_stream_time / cublas_graph_time;
        std::wcout << std::endl;


        CUDA_ERROR(cudaFree(d_r));
        if (ops == ReduceOp::DOT) {
            CUDA_ERROR(cudaFree(d_p));
        }
    }


    std::cout << " ****** Reduce Driver (" << OpToString(ops) << ") with " << num_ops << " operations and "
        << num_nodes << " nodes Stopped ******" << std::endl;
    CUDA_ERROR(cudaProfilerStop());
}


#endif //_REDUCE_DRIVER_