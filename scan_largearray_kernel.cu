#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 32
#define BANK_CONFLICTS (DEFAULT_NUM_ELEMENTS%NUM_BANKS)
#if BANK_CONFLICTS > 0
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void naiveScan(float *g_odata, float *g_idata, int n);
__global__ void sweepScan(float *g_odata, float *g_idata, int n);
__global__ void abcScan(float *g_odata, float *g_idata, int n);
__global__ void BKung(float *g_odata, float *g_idata, int n);

__global__ void simp(float *sums, float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[]; // allocated on invocation
  
  int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("      %d\n", i);
  int offset = 1;
  
  /*
  loading two elements from separate halves of the array, we avoid these bank conflicts.
  Also, to avoid bank conflicts during the tree traversal, we need to 
  add padding to the shared memory array every NUM_BANKS (16) elements
  */
  int ai = tid;
  int bi = tid + (n/2);
  
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  
  temp[ai + bankOffsetA] = g_idata[blockIdx.x * blockDim.x + ai];
  temp[bi + bankOffsetB] = g_idata[blockIdx.x * blockDim.x + bi];
  
  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    //divide n by 2 each time, probably shift so no decimals
    __syncthreads();
    
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai); 
      bi += CONFLICT_FREE_OFFSET(bi);
      
      if (blockIdx.x == 0) {
        printf("%d- %d %d, ai bi %d, %d\n", d, tid, blockIdx.x, ai, bi);
      }
      
      temp[bi] += temp[ai];
    }
    
    
    offset *= 2;
  }
  
  //last element should be sum
  if (tid == blockDim.x - 1) {
    sums[blockIdx.x] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    printf("%d: %d, %d, %d - %f at %d\n", i, blockIdx.x, tid, blockDim.x, temp[(n - 1) + CONFLICT_FREE_OFFSET(n - 1)], (n - 1) + CONFLICT_FREE_OFFSET(n - 1));
  }
  
  __syncthreads();
  
  /*
  if (blockIdx.x == 3) {
    printf("%f : %f\n", sums[blockIdx.x], sums[3]);
  }
  */
  
  if (tid == 0) {
    //printf("%d: %d, %d, %d - %f at %d\n", i, blockIdx.x, tid, blockDim.x, temp[(n - 1) + CONFLICT_FREE_OFFSET(n - 1)], (n - 1) + CONFLICT_FREE_OFFSET(n - 1));
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; // clear the last element
  }
  
  __syncthreads();
  
  for (int d = 1; d < n; d *= 2)  {//traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    
    if (tid < d) {
      int ai = offset*(2*tid+1)-1; 
      int bi = offset*(2*tid+2)-1; 
      ai += CONFLICT_FREE_OFFSET(ai); 
      bi += CONFLICT_FREE_OFFSET(bi);
      
      float t = temp[ai];
      temp[ai] = temp[bi]; 
      temp[bi] += t;
    }
  }
  
  __syncthreads();
  
  //write results to array
  g_odata[blockIdx.x * blockDim.x + ai] = temp[ai + bankOffsetA]; 
  g_odata[blockIdx.x * blockDim.x + bi] = temp[bi + bankOffsetB];
  
  __syncthreads();
  
  /*
  if (i == 6) {
    for(int i = 0; i < 1024; i++) {
      printf("%d : %f\n", i, *(g_odata + i));
    }
  }
  */
  
  
}



__global__ void brent(float *sums, float *g_odata, float *g_idata, int n) {

  extern __shared__ float XY[];
  int index = blockIdx.x * blockDim.x + threadIdx.x;//global thread ID number
  int tid = threadIdx.x;//global thread ID number

  unsigned int i = 2*tid;//1 3 5 7
  if (tid == 0) XY[tid] = 0;//add identity value to first element using the 0th thread
  
  if (i < n){ 
    XY[i+1] = g_idata[blockIdx.x * blockDim.x  + i]; //  1 = 0    3 = 2     5 = 4          7 = 6//copy every other value of the even elements of the input array 
    //XY[i+1] = g_idata[i]; // 2 = 1    4 = 3     6 = 5          8 = 7
  }
  
  if (i+2 < n){ 
    XY[i+2] = g_idata[blockIdx.x * blockDim.x + i + 1]; // 2 = 1    4 = 3     6 = 5          XXXX 8 = 7 //copy every other value of the od elements of the input array
  }
  
  for(unsigned int stride = 1; stride <= n/2; stride*=2){
    __syncthreads();
    unsigned int jj = ((tid+1)*2*stride);
    if(jj < n){
      //if(blockIdx.x == 0 && tid == 0) {
        //printf("XY%d - %d  ---   %f  +  %f\n", stride, jj, XY[jj], XY[jj - stride]);
      //}
      XY[jj] += XY[jj - stride];
      
    }
  }
  
  /*
  if(blockIdx.x == 0 && tid == 255) {
    for(int ii = 0; ii < 256; ii++) {
      printf("XY  ---   %d : %f  |  %f\n", ii, g_idata[blockIdx.x * blockDim.x  + ii], XY[ii]);
    }
  }
  */
  
  //last element should be sum
  if (tid == blockDim.x - 1) {
    sums[blockIdx.x] = XY[n - 1];
    printf("%d: %d, %d, %d - %f at %d\n", index, blockIdx.x, tid, blockDim.x, XY[(n - 1)], (n - 1));
  }
  
  
  for (int stride = n/4; stride > 0; stride /= 2){
    __syncthreads();
    unsigned int ii = ((tid+1)*stride*2);
    if(ii + stride < n){
      XY[ii + stride] += XY[ii];
    }
  }
  
  if (i < n){
    g_odata[blockIdx.x * blockDim.x + i] = XY[i];
  }
  
  if (i+1< n){
    g_odata[blockIdx.x * blockDim.x + i+1] = XY[i+1];
  }

}



__global__ void arbit(float *g_odata, float *incr, int n) {
  //write results to array
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  /*
  if (i == 2) {
    for(int ii = 0; ii < 8; ii++) {
      printf("%d : %f\n", ii, incr[ii]);
    }
  }
  */
  
  __syncthreads();
  
  //printf("%d : %f, %f\n", i, g_odata[i], incr[blockIdx.x]);
  g_odata[i] += incr[blockIdx.x];
  
  __syncthreads();
}




// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *sums, int numElements) {

  //arbitrary length, assuming number of elements is a multiple of B
  //int section_size = 256;
  //int threads_in_block = 512;
  int num_blocks = ceil((float) numElements/ (float) BLOCK_SIZE);
  int num_threads = ceil((float) BLOCK_SIZE/ (float) 2);
  printf("  num blocks    %d  num threads  %d\n", num_blocks, num_threads);
  
  dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(numElements, 1, 1);
  
  dim3 arbit_dimGrid(num_blocks, 1, 1);
	dim3 arbit_dimBlock(num_threads, 1, 1);
 
  dim3 dimG(1, 1, 1);
	dim3 dimB(num_blocks, 1, 1);
 
  dim3 dimGd(num_blocks, 1, 1);
	dim3 dimBk(BLOCK_SIZE, 1, 1);

	// Launch the device computation threads!
  //naiveScan<<<dimGrid, dimBlock, 2*numElements*sizeof(float)>>>(outArray, inArray, numElements);
  //sweepScan<<<dimGrid, dimBlock, 2*numElements*sizeof(float)>>>(outArray, inArray, numElements);
  //abcScan<<<dimGrid, dimBlock, 2*numElements*sizeof(float)>>>(outArray, inArray, numElements);
  //simp<<<arbit_dimGrid, arbit_dimBlock, 2*BLOCK_SIZE*sizeof(float)>>>(sums, outArray, inArray, BLOCK_SIZE);
  //abcScan<<<dimG, dimB, 2*num_blocks*sizeof(float)>>>(sums, sums, num_blocks);
  //arbit<<<dimGd, dimBk>>>(outArray, sums, numElements);
  
  //BKung<<<dimGrid, dimBlock, sizeof(float)*numElements>>>(outArray, inArray, numElements);
  brent<<<arbit_dimGrid, arbit_dimBlock, sizeof(float)*BLOCK_SIZE>>>(sums, outArray, inArray, BLOCK_SIZE);
  BKung<<<dimG, dimB, sizeof(float)*num_blocks>>>(sums, sums, num_blocks);
  arbit<<<dimGd, dimBk>>>(outArray, sums, numElements);
}
// **===-----------------------------------------------------------===**


__global__ void BKung(float *g_odata, float *g_idata, int n) {

  extern __shared__ float XY[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;//global thread ID number

  
  unsigned int i = 2*tid;//1 3 5 7
  if (tid == 0) XY[tid] = 0;//add identity value to first element using the 0th thread
  
  if (i < n){ 
    XY[i+1] = g_idata[i]; //  1 = 0    3 = 2     5 = 4          7 = 6//copy every other value of the even elements of the input array 
    //XY[i+1] = g_idata[i]; // 2 = 1    4 = 3     6 = 5          8 = 7
  }
  if (i+2 < n){ 
    XY[i+2] = g_idata[i + 1]; // 2 = 1    4 = 3     6 = 5          XXXX 8 = 7 //copy every other value of the od elements of the input array
  }
  
  for(unsigned int stride = 1; stride <= n/2; stride*=2){
    __syncthreads();
    unsigned int index = ((tid+1)*2*stride);
    if(index < n){
      XY[index] += XY[index-stride];
    }
  }
  for (int stride = n/4; stride > 0; stride /= 2){
    __syncthreads();
    unsigned int index = ((tid+1)*stride*2);
    if(index + stride < n){
      XY[index + stride] += XY[index];
    }
  }
  if (i < n){
    g_odata[i] = XY[i];
  }
  if (i+1< n){
    g_odata[i+1] = XY[i+1];
  }
  
  if (tid == 3) {
    for(int i = 0; i < 2; i++) {
      printf("sums: %d : %f\n", i, *(g_odata + i));
    }
  }

}


__global__ void abcScan(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[]; // allocated on invocation
  
  int tid = threadIdx.x;
  int offset = 1;
  
  /*
  loading two elements from separate halves of the array, we avoid these bank conflicts.
  Also, to avoid bank conflicts during the tree traversal, we need to 
  add padding to the shared memory array every NUM_BANKS (16) elements
  */
  int ai = tid;
  int bi = tid + (n/2);
  
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  
  temp[ai + bankOffsetA] = g_idata[ai];
  temp[bi + bankOffsetB] = g_idata[bi];
  
  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    //divide n by 2 each time, probably shift so no decimals
    __syncthreads();
    
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai); 
      bi += CONFLICT_FREE_OFFSET(bi);
      
      temp[bi] += temp[ai];
    }
    
    offset *= 2;
  }
  
  if (tid == 0) {
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; // clear the last element
  }
  
  for (int d = 1; d < n; d *= 2)  {//traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    
    if (tid < d) {
      int ai = offset*(2*tid+1)-1; 
      int bi = offset*(2*tid+2)-1; 
      ai += CONFLICT_FREE_OFFSET(ai); 
      bi += CONFLICT_FREE_OFFSET(bi);
      
      float t = temp[ai];
      temp[ai] = temp[bi]; 
      temp[bi] += t;
    }
  }
  
  __syncthreads();
  
  //write results to array
  g_odata[ai] = temp[ai + bankOffsetA]; 
  g_odata[bi] = temp[bi + bankOffsetB];
  
  __syncthreads();
  
  //if (tid == 3) {
    //for(int i = 0; i < 4; i++) {
      //printf("%d : %f\n", i, *(g_odata + i));
    //}
  //}
}





__global__ void sweepScan(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[]; // allocated on invocation
  
  int tid = threadIdx.x;
  int offset = 1;
  
  temp[2*tid] = g_idata[2*tid]; //load 1st and second element into shared mem
  temp[2*tid+1] = g_idata[2*tid+1];
  
  for (int d = n >> 1; d > 0; d >>= 1) {  // build sum in place up the tree
    //divide n by 2 each time, probably shift so no decimals
    __syncthreads();
    
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      
      temp[bi] += temp[ai];
    }
    
    offset *= 2;
  }
  
  if (tid == 0) {
    temp[n-1] = 0; // clear the last element
  }
  
  for (int d = 1; d < n; d *= 2)  {//traverse down tree & build scan
    offset >>= 1;
    __syncthreads();
    
    if (tid < d) {
      int ai = offset*(2*tid+1)-1; 
      int bi = offset*(2*tid+2)-1; 
      
      float t = temp[ai];
      temp[ai] = temp[bi]; 
      temp[bi] += t;
    }
  }
  
  __syncthreads();
  
  g_odata[2*tid] = temp[2*tid];  // write results to device memory
  g_odata[2*tid+1] = temp[2*tid+1];
}



__global__ void naiveScan(float *g_odata, float *g_idata, int n) {

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
  
  for (int offset = 1; offset < n; offset *= 2) { 
    buffer1 = 1 - buffer1; // swap double buffer indices //b1 = 1 - 0 = 1, b1 = 1 - 1 = 0
    buffer2 = 1 - buffer1;                                //b2 = 1 - 1 = 0, b2 = 1 - 0 = 1
    
    if (tid >= offset) { 
      temp[buffer1*n+tid] = temp[buffer2*n+tid - offset] + temp[buffer2*n+tid]; 
    } else {
      temp[buffer1*n+tid] = temp[buffer2*n+tid]; 
    }
    
    __syncthreads();
  }
  
  g_odata[tid] = temp[buffer1*n+tid]; // write output
  
}

#endif // _PRESCAN_CU_
