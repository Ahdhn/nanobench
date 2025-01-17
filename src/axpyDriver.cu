#ifndef _AXPY_DRIVER_
#define _AXPY_DRIVER_


#include "util.cu"
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cuda_profiler_api.h>

inline bool moveAndCompare(const int N, double* d_p, double* h_p_gold) {

    std::vector<double> h_p_res(N, 0);
    CUDA_ERROR(cudaMemcpy(h_p_res.data(), d_p, N * sizeof(double), cudaMemcpyDeviceToHost));
    return compare_arr(h_p_gold, h_p_res.data(), N);
}

/**
* memcpy_kernel()
*/
__global__ void static memcpy_kernel(double* d_dest, const double* d_src, uint32_t length) {
    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length) {
        d_dest[i] = d_src[i];
        i += stride;
    }
}

/**
* handmadeDaxpy()
*/
__global__ void static handmadeDaxpy(const int N, double* d_r, double* d_p, double alpha) {

    const uint32_t stride = blockDim.x * gridDim.x;
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N) {
        double p = d_p[i];
        double r = d_r[i];
        p *= alpha;
        p += r;
        d_p[i] = p;
        i += stride;
    }
}


/**
* handmadeDaxpyGraph()
*/
inline float handmadeDaxpyGraph(const int N, double* d_r, double* d_p, const int num_ops,
                                double* h_p_gold, const int sm_count) {
    //return the time in float         
    const double alpha = 1.0;
    const int threads = 256;
    //const int blocks = (N + threads - 1) / threads;
    int num_blocks_per_sm = 0;
    CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
        (void*)handmadeDaxpy, threads, 0));
    const int blocks = num_blocks_per_sm * sm_count;

    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));
    
    cudaGraph_t graph;
    CUDA_ERROR(cudaGraphCreate(&graph, 0));
    cudaKernelNodeParams kernel_node_params = { 0 };    
    

    void* kernel_args[4] = { (void*)&N, (void*)&d_r, (void*)&d_p, (void*)&alpha };
    kernel_node_params.func = (void*)handmadeDaxpy;
    kernel_node_params.gridDim = dim3(blocks, 1, 1);
    kernel_node_params.blockDim = dim3(threads, 1, 1);
    kernel_node_params.sharedMemBytes = 0;
    kernel_node_params.kernelParams = (void**)kernel_args;
    kernel_node_params.extra = NULL;    

    cudaGraphNode_t kernel_node;
    CUDA_ERROR(cudaGraphAddKernelNode(&kernel_node, graph, NULL,
        0, &kernel_node_params));    

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(graph, nodes, &generated_num_nodes));
    if (generated_num_nodes != 1) {
        fprintf(stderr, "handmadeDaxpyGraph():: CUDA Graph has generated %d but the expected is %d",
            static_cast<int>(generated_num_nodes), 1);
        exit(EXIT_FAILURE);
    }

    cudaGraphExec_t exec_graph;
    CUDA_ERROR(cudaGraphInstantiate(&exec_graph, graph, NULL, NULL, 0));

    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, stream));

    for (int iter = 0; iter < num_ops; ++iter) {
        CUDA_ERROR(cudaGraphLaunch(exec_graph, stream));        
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    CUDA_ERROR(cudaEventRecord(stop, stream));
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaStreamDestroy(stream));

    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));


    if (!moveAndCompare(N, d_p, h_p_gold)) {
        std::cout << " handmadeDaxpyGraph() failed with N = " << N << std::endl;
        exit(EXIT_FAILURE);
    }
    return time / num_ops;
}

/** 
* handmadeDaxpyStream()
*/
inline float handmadeDaxpyStream(const int N, double* d_r, double* d_p, const int num_ops,
    double* h_p_gold, const int sm_count) {
    //return the time in float 
    const double alpha = 1.0;
    const int threads = 256;
    //const int blocks = (N + threads - 1) / threads;
    int num_blocks_per_sm = 0;
    CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
        (void*)handmadeDaxpy, threads, 0));
    const int blocks = num_blocks_per_sm * sm_count;

    cudaStream_t stream;
    CUDA_ERROR(cudaStreamCreate(&stream));
    
    cudaEvent_t start, stop;
    CUDA_ERROR(cudaEventCreate(&start));
    CUDA_ERROR(cudaEventCreate(&stop));
    CUDA_ERROR(cudaEventRecord(start, stream));
    for (int iter = 0; iter < num_ops; ++iter) {
        handmadeDaxpy <<<blocks, threads,0, stream >>>(N, d_r, d_p, alpha);
        CUDA_ERROR(cudaStreamSynchronize(stream));
    }

    CUDA_ERROR(cudaEventRecord(stop, stream));
    CUDA_ERROR(cudaEventSynchronize(stop));
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());
    CUDA_ERROR(cudaStreamDestroy(stream));
    
    float time = 0.0f;//ms
    CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));


    if (!moveAndCompare(N, d_p, h_p_gold)) {
        std::cout << " handmadeDaxpyStream() failed with N = " << N << std::endl;
        exit(EXIT_FAILURE);
    }
    return time / num_ops;
}

/**
* cublasDaxpyStream()
*/
inline float cublasDaxpyStream(const int N, double* d_r, double* d_p, const int num_ops,
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
        std::cout << " cublasDaxpyStream() failed with N = " << N << std::endl;
        exit(EXIT_FAILURE);
    }



    return time/ num_ops;

}

/**
* cublasDaxpyGraph()
*/
inline float cublasDaxpyGraph(const int N, double*d_r, double *d_p, const int num_ops,
    double* h_p_gold) {
    //return the time in float     
    
    double alpha = 1.0;    
    cudaGraph_t cuda_graph;
    cudaStream_t capture_stream;
    cublasHandle_t cublas_handle = 0;
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    CUDA_ERROR(cudaStreamCreate(&capture_stream));
    CUDA_ERROR(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    CUBLAS_ERROR(cublasSetStream(cublas_handle, capture_stream));
    CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    
    CUBLAS_ERROR(cublasDaxpy(cublas_handle, N, &alpha, d_r, 1, d_p, 1));    
    CUDA_ERROR(cudaStreamEndCapture(capture_stream, &cuda_graph));

    cudaGraphNode_t* nodes = NULL;
    size_t generated_num_nodes = 0;
    CUDA_ERROR(cudaGraphGetNodes(cuda_graph, nodes, &generated_num_nodes));
    if (generated_num_nodes != 1) {
        fprintf(stderr, "cublasDaxpyGraph():: CUDA Graph has generated %d but the expected is %d",        
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

    if (!moveAndCompare(N, d_p, h_p_gold)) {
        fprintf(stderr, "cublasDaxpyGraph():: failed with N= %d", N);                
        exit(EXIT_FAILURE);
    }

    return time/ num_ops;
}

/**
* benchDriver()
*/
inline void axpyDriver(const int num_ops, const int start, const int end, 
    const double max_bandwidth, const int sm_count) {

    CUDA_ERROR(cudaProfilerStart());
    std::cout << " ****** AXPY Driver with " << num_ops << " operations Started ******" << std::endl;
    const char separator = ' ';
    const int numWidth = 20;
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Exp (2^x)";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Size";
    //std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Theoretical Time";    
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "CUBLAS GraphTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "CUBLAS StreamTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "HANDMADE GraphTime";
    std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "HANDMADE StreamTime";
    //std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << "Speedup";
    std::cout << std::endl << std::endl;

    for (int exp = start; exp <= end; ++exp) {
        int N = 1 << exp;                
        const double theoretical_time = (N / 10e6) * (24 / max_bandwidth);        
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

        //this is just for timing (2 read and 1 write)
        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));        

        //Launch cublas graph
        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float cublas_graph_time = cublasDaxpyGraph(N, d_r, d_p, num_ops, h_p_gold.data());

        //Launch cublas streams 
        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float cublas_stream_time = cublasDaxpyStream(N, d_r, d_p, num_ops, h_p_gold.data());
        
        //Launch handmade graph
        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float handmade_graph_time = handmadeDaxpyGraph(N, d_r, d_p, num_ops, h_p_gold.data(), sm_count);

        //Launch handmade stream
        CUDA_ERROR(cudaMemcpy(d_r, h_r.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_p, h_p.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        float handmade_stream_time = handmadeDaxpyStream(N, d_r, d_p, num_ops, h_p_gold.data(), sm_count);

        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << exp;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << N;
        //std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << theoretical_time;        
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_graph_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_stream_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << handmade_graph_time;
        std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << handmade_stream_time;
        //std::cout << std::left << std::setw(numWidth) << std::setfill(separator) << cublas_stream_time/ cublas_graph_time;
        std::wcout << std::endl;
        

        CUDA_ERROR(cudaFree(d_r));
        CUDA_ERROR(cudaFree(d_p));       
    }
    std::cout << " ****** AXPY Driver with " << num_ops << " operations Stopped ******" << std::endl;    

    CUDA_ERROR(cudaProfilerStop());
}



#endif //_AXPY_DRIVER_