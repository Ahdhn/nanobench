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
	
	/*axpyDriver(200, 1, 1, 24);
	axpyDriver(200, 5, 1, 24);
	axpyDriver(200, 10, 1, 24);
	axpyDriver(200, 50, 1, 24);
	axpyDriver(200, 100, 1, 24);
	axpyDriver(200, 200, 1, 24);*/
	//axpyDriver(200, 5, 1, 28, max_bandwidth);

	reduceDriver(1,1,1, 30, max_bandwidth, ReduceOp::NORM2);

	/*for (int n = 1; n < 28; ++n) {
		axpyDriver(n, 1, 1, 28);
	}*/


	return 0;
}
