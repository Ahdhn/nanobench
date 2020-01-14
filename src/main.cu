#include <stdio.h>
#include <stdint.h>

#include "bencher.cu"
#include "cuda_query.h"

int main(int argc, char** argv) {		
	uint32_t device_id = 0;
	CUDA_ERROR(cudaSetDevice(device_id));
	cuda_query(device_id);
	
	benchDriver(50, 1, 28);


	return 0;
}
