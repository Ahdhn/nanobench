#ifndef _CUDA_QUERY_
#define _CUDA_QUERY_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "util.cu"

int _ConvertSMVer2Cores(int major, int minor){
	//Taken from Nvidia helper_cuda.h to get the number of SM and cuda cores 
	//Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class
		{ 0x72, 64 },
		{ 0x75, 64 },
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1){
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)){
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

	
cudaDeviceProp cuda_query(const int dev){

	//Various query about the device we are using 	
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        PRINT_ERROR("cuda_query() device count = 0 i.e., there is not"
            " a CUDA-supported GPU!!!");
    }

	cudaSetDevice(dev);
	cudaDeviceProp devProp;

	CUDA_ERROR(cudaGetDeviceProperties(&devProp, dev));

	printf("\n Total number of device: %d", deviceCount);
	printf("\n Using device Number: %d", dev);
	printf("\n Device name: %s", devProp.name);

	printf("\n Compute Capability: %d.%d", (int)devProp.major, (int)devProp.minor);

	char msg[256];

#ifdef _WIN32
	sprintf_s(msg, "\n Total amount of global memory (MB): %.0f", (float)devProp.totalGlobalMem / 1048576.0f);
#else 
	sprintf(msg, "\n  Total amount of global memory: %.0f", (float)devProp.totalGlobalMem / 1048576.0f);
#endif

	printf("%s", msg);

	printf("\n (%2d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores",
		devProp.multiProcessorCount,
		_ConvertSMVer2Cores(devProp.major, devProp.minor),
		_ConvertSMVer2Cores(devProp.major, devProp.minor) * devProp.multiProcessorCount);


	printf("\n GPU Max Clock rate: %.0f MHz (%0.2f GHz)", devProp.clockRate * 1e-3f, devProp.clockRate * 1e-6f);
	printf("\n Memory Clock rate: %.0f Mhz", devProp.memoryClockRate * 1e-3f);
	printf("\n Memory Bus Width:  %d-bit", devProp.memoryBusWidth);
	const double maxBW = 2.0 * devProp.memoryClockRate*(devProp.memoryBusWidth / 8.0) / 1.0E6;
	printf("\n Peak Memory Bandwidth: %f(GB/s)", maxBW);
	printf("\n Kernels compiled for compute capability: %d\n\n", cuda_arch());

	return devProp;
}







#endif /*_CUDA_QUERY_*/