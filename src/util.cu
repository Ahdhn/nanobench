#ifndef  _UTIL_
#define  _UTIL_

#include <cublas_v2.h>
#include <stdio.h>
#include <iostream>

//********************** CUDA_ERROR
inline void HandleError(cudaError_t err, const char *file, int line) {
	//Error handling micro, wrap it around function whenever possible
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);

#ifdef _WIN32
		system("pause");
#else
		exit(EXIT_FAILURE);
#endif
	}
}
#define CUDA_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
//******************************************************************************


//********************** CUBLAS_ERROR
static const char* cudaGetCUBLASErrorName(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}


template <typename T>
inline void HandleCUBLASError(T result, char const* const func, 
	const char* const file, int const line) {
	
	if (result) {
		printf("CUBLAS error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), cudaGetCUBLASErrorName(result), func);

#ifdef _WIN32
		system("pause");
#else
		exit(EXIT_FAILURE);
#endif
}
}

#define CUBLAS_ERROR(err) HandleCUBLASError((err), #err, __FILE__, __LINE__)
//******************************************************************************

//********************** PRINT_ERROR
#ifndef PRINT_ERROR
#include <stdio.h>
#include <string>
inline void Err(std::string err_line, const char* file, int line,
	bool stop = 1) {
	//Display the err_line
	printf("Error::%s \n Error generated in %s at line %d\n", err_line.c_str(), file, line);

	if (stop) {
#ifdef _WIN32
		system("pause");
#else
		exit(EXIT_FAILURE);
#endif
	}

}
#define PRINT_ERROR( err_line ) (Err( err_line, __FILE__, __LINE__ ))
#endif
//******************************************************************************


//********************** Get Cuda Arch
__global__ void static get_cude_arch_k(int*d_arch){

#if defined(__CUDA_ARCH__)
	*d_arch = __CUDA_ARCH__;
#else
	*d_arch = 0;	
#endif

}
inline int cuda_arch(){
	int*d_arch = 0;
	CUDA_ERROR(cudaMalloc((void**)&d_arch, sizeof(int)));
	get_cude_arch_k << < 1, 1 >> >(d_arch);
	int h_arch = 0;
	CUDA_ERROR(cudaMemcpy(&h_arch, d_arch, sizeof(int), cudaMemcpyDeviceToHost));
	cudaFree(d_arch);
	return h_arch;
}
//******************************************************************************

//********************** compare
template<typename T, typename dataT>
bool compare_arr(const dataT* gold, const dataT* arr, const T size,
	const bool verbose = false, const dataT tol = 10E-5) {

	bool result = true;
	for (T i = 0; i < size; i++) {
		if (abs(double(gold[i]) - double(arr[i])) > tol) {
			if (verbose) {
				std::cout << " compare array mismatch at [" << i
					<< "]  gold = " << gold[i] << ", arr=" << arr[i]
					<< std::endl;
				result = false;
			}
			else {
				//it is not verbose, don't bother running through all entires
				return false;
			}
		}
	}
	return result;
}

//******************************************************************************
#endif // ! _UTIL_