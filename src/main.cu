#include <stdio.h>
#include <stdint.h>

#include "reduceDriver.cu"
#include "axpyDriver.cu"
#include "cuda_query.h"

int main(int argc, char** argv) {		
	uint32_t device_id = 0;
	double max_bandwidth = 0;
	CUDA_ERROR(cudaSetDevice(device_id));
	cuda_query(device_id, max_bandwidth);
	
	axpyDriver(200, 1, 30, max_bandwidth);
	reduceDriver(200, 1, 30, max_bandwidth, ReduceOp::DOT);
	reduceDriver(200, 1, 30, max_bandwidth, ReduceOp::NORM2);


	return 0;
}

