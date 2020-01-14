#include <stdio.h>
#include <stdint.h>

#include "bencher.cu"
#include "cuda_query.h"

int main(int argc, char** argv) {		
	uint32_t device_id = 3;
	CUDA_ERROR(cudaSetDevice(device_id));
	cuda_query(device_id);
	
	benchDriver(200, 1, 1, 24);
	benchDriver(200, 5, 1, 24);
	benchDriver(200, 10, 1, 24);
	benchDriver(200, 50, 1, 24);
	benchDriver(200, 100, 1, 24);
	benchDriver(200, 200, 1, 24);

	/*for(int n = 1; n<50; ++n){
		benchDriver(200, n, 1, 24);
	}*/
	return 0;
}
