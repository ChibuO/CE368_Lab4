#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256
#define double_num 16

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void scan(float *g_odata, float *g_idata, int n) {

  extern __shared__ float temp[]; // allocated on invocation
  
  int tid = threadIdx.x;
  int buffer1 = 0;
  int buffer2 = 1;
  
  // load input into shared memory. 
  // This is exclusive scan, so shift right by one and set first element to 0

  if (tid > 0) {
    temp[buffer1*n + tid] = g_idata[tid-1];
  } else {
    temp[buffer1*n + tid] = 0;
  }
  __syncthreads(); 
  
  /*
  for (int offset = 1; offset < n; offset *= 2) { 
    buffer1 = 1 - buffer1; // swap double buffer indices //b1 = 1 - 0 = 1, b1 = 1 - 1 = 0
    buffer2 = 1 - buffer1;                                //b2 = 1 - 1 = 0, b2 = 1 - 0 = 1
    
    if (tid >= offset) {
      printf("hey\n");
      //temp[buffer1*n+tid] += temp[buffer2*n+tid - offset]; 
    } else {
      printf("bye\n");
      //temp[buffer1*n+tid] = temp[buffer2*n+tid]; 
    }
    
    __syncthreads(); 
  }
  */
  
  //g_odata[tid] = temp[buffer1*n+tid+1]; // write output 1???
  
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements) {
  
  dim3 dimGrid(65536, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 4, 1);

	// Launch the device computation threads!
  scan<<<dimGrid, dimBlock, numElements*2*sizeof(float)>>>(outArray, inArray, numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
