#ifndef _REDUCE_DRIVER_
#define _REDUCE_DRIVER_

#include "util.cu"
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cub\cub.cuh>

enum class ReduceOp
{
    DOT = 0,
    NORM2 = 1,
};
inline std::string OpToString(const ReduceOp ops) {
    return (ops == ReduceOp::DOT) ? "DOT" : "NORM2";
}

template <int blockThreads>
__device__ __forceinline__ void cubBlockReducer(const double&thread_data,
    double* d_per_block_res) {

    typedef cub::BlockReduce<double, blockThreads>BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_total = BlockReduce(temp_storage).Sum(thread_data);
    if (threadIdx.x == 0) {
        d_per_block_res[blockIdx.x] = block_total;
    }
}
/**
* blockNorm2()
*/
template<int blockThreads>
__global__ static void blockNorm2(const int N, const double* d_r,
    double* d_per_block_res) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double thread_data = 0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        double r = d_r[i];
        thread_data += r * r;
    }
    __syncthreads();
    cubBlockReducer<blockThreads>(thread_data, d_per_block_res);
}

/**
* blockDot()
*/
template<int blockThreads>
__global__ static void blockDot(const int N, const double*d_r, const double*d_p, 
    double*d_per_block_res){

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    double thread_data = 0;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x){
        double p = d_p[i];
        double r = d_r[i];        
        thread_data += p * r;        
    }
    __syncthreads();    
    cubBlockReducer<blockThreads>(thread_data, d_per_block_res);
}

/**
* sqrt()
*/
__global__ static void sqrt(double* d_res) {
    //launch only one thread to do this 
    d_res[0] = sqrt(d_res[0]);
}

/**
* handmadeReduceGraph()
*/
inline float handmadeReduceGraph(const int N, double *d_r, double*d_p, const ReduceOp ops, 
    const int num_ops, const double gold) {

    //return the time in float 
    const int threads = 256;
    //stride is number of hops a the block will make
    const int num_hops = 10;
    const int stride = (N + num_hops - 1) / num_hops;
    const int blocks = (stride + threads - 1) / threads;
    

    double* d_per_block_res(NULL);
    CUDA_ERROR(cudaMalloc((void**)&d_per_block_res, blocks * sizeof(double)));
    double* d_res(NULL);
    CUDA_ERROR(cudaMalloc((void**)&d_res, sizeof(double)));

    void* d_cub_temp_storage = NULL;
    size_t cub_temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_cub_temp_storage, cub_temp_storage_bytes, 
        d_per_block_res, d_res, blocks);    
    CUDA_ERROR(cudaMalloc(&d_cub_temp_storage, cub_temp_storage_bytes));

    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));
    cudaGraph_t graph;    
    CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    switch (ops)
    {
    case ReduceOp::DOT: {
        blockDot<threads> <<< blocks, threads, 0, stream >>> (N, d_r, d_p, d_per_block_res);
        break;
    }
    case ReduceOp::NORM2: {
        blockNorm2<threads> <<< blocks, threads, 0, stream >>> (N, d_r, d_per_block_res);
        break;
    }
    default:
        fprintf(stderr, "handmadeReduceGraph():: unsupported operation!!");
        exit(EXIT_FAILURE);        
    }
    cub::DeviceReduce::Sum(d_cub_temp_storage, cub_temp_storage_bytes, d_per_block_res,
        d_res, blocks, stream);   
    if (ops == ReduceOp::NORM2) {
        sqrt <<< 1, 1, 0, stream >>>(d_res);
    }
    CUDA_ERROR(cudaStreamEndCapture(stream, &graph));

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(graph, nodes, &generated_num_nodes));
    /*if (generated_num_nodes != 2) {
        fprintf(stderr, "handmadeReduceGraph():: CUDA Graph has generated %d but the expected is %d",
            static_cast<int>(generated_num_nodes), 1);
        exit(EXIT_FAILURE);
    }*/
    
    cudaGraphExec_t cuda_graph_exec;
    CUDA_ERROR(cudaGraphInstantiate(&cuda_graph_exec, graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, stream));
    for (int iter = 0; iter < num_ops; ++iter) {
        CUDA_ERROR(cudaGraphLaunch(cuda_graph_exec, stream));
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    CUDA_ERROR(cudaEventRecord(stop, stream));
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaGraphExecDestroy(cuda_graph_exec));
    CUDA_ERROR(cudaGraphDestroy(graph));
    CUDA_ERROR(cudaStreamDestroy(stream));    

    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

    double h_res(0);
    CUDA_ERROR(cudaMemcpy((void*)&h_res, (void*)d_res, sizeof(double), cudaMemcpyDeviceToHost));

    if (std::abs(h_res - gold) > 0.00001) {
        fprintf(stderr, "handmadeReduceGraph():: failed with N= %d", N);
        exit(EXIT_FAILURE);
    }

    CUDA_ERROR(cudaFree(d_per_block_res));
    CUDA_ERROR(cudaFree(d_res));
    CUDA_ERROR(cudaFree(d_cub_temp_storage));

    return time / num_ops;
}

/**
* cublasReduceStream()
*/
inline float cublasReduceStream(const int N, double* d_r, double* d_p,
    const ReduceOp ops, const int num_ops, const double gold) {
        
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
            fprintf(stderr, "cublasReduceStream():: unsupported operation!!");
            exit(EXIT_FAILURE);            
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
        fprintf(stderr, "cublasReduceStream():: failed with N= %d", N);
        exit(EXIT_FAILURE);
    }

    CUDA_ERROR(cudaFree(d_res));


    return time / num_ops;
}


/**
* cublasReduceGraph()
*/
inline float cublasReduceGraph(const int N, double*d_r, double*d_p, const ReduceOp ops,
    const int num_ops, const double gold) {
    
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
        fprintf(stderr, "cublasReduceGraph():: unsupported operation!!");
        exit(EXIT_FAILURE);
        break;
    }

    CUDA_ERROR(cudaStreamEndCapture(capture_stream, &cuda_graph));

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(cuda_graph, nodes, &generated_num_nodes));

    //This factor because different reduction ops (dot vs. norm) in CUBLAS produces
    //different number of nodes 
    int factor = (ops == ReduceOp::DOT) ? 2 : 4;

    if (generated_num_nodes != factor) {
        fprintf(stderr, "cublasReduceGraph():: CUDA Graph has generated %d but the expected is %d",
            static_cast<int>(generated_num_nodes), 1);
        exit(EXIT_FAILURE);
    }
    cudaGraphExec_t cuda_graph_exec;
    CUDA_ERROR(cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, capture_stream));
    for (int iter = 0; iter < num_ops; ++iter) {
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

inline void reduceDriver(const int num_ops, const int start, const int end,  
    const double max_bandwidth, const ReduceOp ops) {
    //
    CUDA_ERROR(cudaProfilerStart());
    std::cout << " ****** Reduce Driver ("<< OpToString (ops) <<") with " << 
        num_ops << " operations Started ******" << std::endl;
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
                fprintf(stderr, "reduceDriver():: unsupported operation!!");
                exit(EXIT_FAILURE);                
            }            
        }
        if (ops == ReduceOp::NORM2) {
            gold = std::sqrt(gold);
        }

        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        if (ops == ReduceOp::DOT) {
            CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        }

        //float cublas_graph_time = cublasReduceGraph(N, d_r, d_p, ops, num_ops, gold);        
        float cublas_stream_time = cublasReduceStream(N, d_r, d_p, ops, num_ops, gold);
        float handmade_graph_time = handmadeReduceGraph(N, d_r, d_p, ops, num_ops, gold);
        float handmade_stream_time = 0;

        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << exp;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << N;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << theoretical_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << practical_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "???";//cublas_graph_time;
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


    std::cout << " ****** Reduce Driver (" << OpToString(ops) << ") with " 
        << num_ops << " operations Stopped ******" << std::endl;
    CUDA_ERROR(cudaProfilerStop());
}


#endif //_REDUCE_DRIVER_