#ifndef _CUGRAPH_
#define _CUGRAPH_


#include "util.cu"
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

inline bool moveAndCompare(const int N, double* d_p, double* h_p_gold) {

    std::vector<double> h_p_res(N, 0);
    CUDA_ERROR(cudaMemcpy(h_p_res.data(), d_p, N * sizeof(double), cudaMemcpyDeviceToHost));
    return compare_arr(h_p_gold, h_p_res.data(), N);
}

inline float DaxpyStream(const int N, double* d_r, double* d_p, const int num_ops,
     double* h_p_gold) {
     //return the time in float 
    double alpha = 1.0;        
    cudaStream_t cublas_stream;
    CUDA_ERROR(cudaStreamCreate(&cublas_stream));
    cublasHandle_t cublas_handle = 0;

    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_ERROR(cublasSetStream(cublas_handle, cublas_stream));

    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, cublas_stream));
    for (int iter = 0; iter < num_ops; ++iter) {
        CUBLAS_ERROR(cublasDaxpy(cublas_handle, N, &alpha, d_r, 1, d_p, 1));    
        CUDA_ERROR(cudaStreamSynchronize(cublas_stream));
    }

    CUDA_ERROR(cudaEventRecord(stop, cublas_stream));
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());    
    CUDA_ERROR(cudaStreamDestroy(cublas_stream));
    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
       

    if (!moveAndCompare(N, d_p, h_p_gold)) {
        std::cout << " DaxpyStream() failed with N = " << N << std::endl;
        exit(EXIT_FAILURE);
    }



    return time/ num_ops;

}

inline float DaxpyGraph(const int N, double*d_r, double *d_p, const int num_ops,
                        const int num_nodes, double* h_p_gold) {
    //return the time in float 
    if(num_ops % num_nodes != 0){
        fprintf(stderr, "DaxpyGraph():: num_ops shoudld be divisible by num_nodes");        
        exit(EXIT_FAILURE);
    }
    
    double alpha = 1.0;
    int num_runs = num_ops/num_nodes;
    cudaGraph_t cuda_graph;
    cudaStream_t capture_stream;
    cublasHandle_t cublas_handle = 0;
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CUDA_ERROR(cudaStreamCreate(&capture_stream));
    CUDA_ERROR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    CUBLAS_ERROR(cublasSetStream(cublas_handle, capture_stream));
    CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    for(int n=0;n<num_nodes;++n){
        CUBLAS_ERROR(cublasDaxpy(cublas_handle, N, &alpha, d_r, 1, d_p, 1));
    }
    CUDA_ERROR(cudaStreamEndCapture(capture_stream, &cuda_graph));

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(cuda_graph, nodes, &generated_num_nodes));
    if (generated_num_nodes != num_nodes) {
        fprintf(stderr, "DaxpyGraph():: CUDA Graph has generated %d but the input is %d", 
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

    if (!moveAndCompare(N, d_p, h_p_gold)) {
        fprintf(stderr, "DaxpyGraph():: failed with N= %d", N);                
        exit(EXIT_FAILURE);
    }

    return time/ num_ops;
}

inline void benchDriver(const int num_ops, const int num_nodes,
                        const int start, const int end) {

    std::cout << " ****** Bench Driver with "<<  num_nodes<< " nodes Started ******" << std::endl;   
    const char separator = ' ';
    const int numWidth = 15;
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Exp";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Size";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "GraphTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "StreamTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Speedup" << std::endl << std::endl;
    for (int exp = start; exp <= end; ++exp) {
        int N = 1 << exp;        
        double* d_r, * d_p;
        CUDA_ERROR(cudaMalloc((void**)&d_r, N * sizeof(double)));
        CUDA_ERROR(cudaMalloc((void**)&d_p, N * sizeof(double)));

        std::vector<double> h_r(N, 0);
        std::vector<double> h_p(N, 0);
        
        std::generate(h_r.begin(), h_r.end(), []() {return double(rand()) / double(RAND_MAX); });
        std::generate(h_p.begin(), h_p.end(), []() {return double(rand()) / double(RAND_MAX); });
        
        std::vector<double> h_p_gold(h_p);
        for (int iter = 0; iter < num_ops; ++iter) {
            for (int i = 0; i < N; ++i) {
                h_p_gold[i] = h_r[i] + h_p_gold[i];
            }
        }

        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float graph_time = DaxpyGraph(N, d_r, d_p, num_ops, 10, h_p_gold.data());

        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float stream_time = DaxpyStream(N, d_r, d_p, num_ops, h_p_gold.data());

        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << exp;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << N;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << graph_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << stream_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << stream_time/graph_time << std::endl;        
        

        CUDA_ERROR(cudaFree(d_r));
        CUDA_ERROR(cudaFree(d_p));
    }
    std::cout << " ****** Bench Driver with "<<  num_nodes<< " nodes Started ******" << std::endl;   
}



#endif // ! _CUGRAPH_