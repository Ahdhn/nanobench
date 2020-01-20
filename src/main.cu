#include <stdio.h>
#include <stdint.h>

#include "reduceDriver.cu"
#include "axpyDriver.cu"
#include "cuda_query.h"

int main(int argc, char** argv) {		
	uint32_t device_id = 0;
	double max_bandwidth = 0;	
	CUDA_ERROR(cudaSetDevice(device_id));
	cudaDeviceProp dev_props = cuda_query(device_id, max_bandwidth);
	int sm_count = dev_props.multiProcessorCount;

	axpyDriver(200, 1, 24, max_bandwidth, sm_count);
	reduceDriver(200, 1, 24, max_bandwidth, ReduceOp::DOT, sm_count);
	reduceDriver(200, 1, 24, max_bandwidth, ReduceOp::NORM2, sm_count);


	return 0;
}

