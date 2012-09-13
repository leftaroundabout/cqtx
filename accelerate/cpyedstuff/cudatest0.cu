// incrementArray.cu
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>



__global__ void calcfouriercomponent(float *a, float *res, int N, float omega) {
  int j = blockIdx.x*blockDim.x + threadIdx.x;
  if (j<N) res[j] = a[j]*sin( (float)j * omega );
}


int main(void) {
  float *a_h, *b_h, *c_hh, *c_dh;

  float *a_d, *b_d;

  int i,j, N = 128, M = 64;
  size_t size = N*sizeof(float);
  size_t sizepr = M*sizeof(float);

  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);
  c_hh = (float *)malloc(sizepr);
  c_dh = (float *)malloc(sizepr);

  cudaMalloc((void **) &a_d, size);
  cudaMalloc((void **) &b_d, size);

  for (i=0; i<N; ++i) a_h[i] = sin ( (float)i/14 );
  for (i=0; i<M; ++i) c_dh[i] = c_hh[i] = 0;

  cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);


  int blockSize = 4;
  int nBlocks = N/blockSize + (N%blockSize == 0?0:1);

  float omega, bvh;
  for (i=0; i<M; ++i) {
    omega = 31.4 * ((float)i)/((float)N * N);
    calcfouriercomponent <<< nBlocks, blockSize >>> ( a_d, b_d, N, omega );
    cudaMemcpy(b_h, b_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
    for (j=0; j<N; ++j) {
      bvh = a_h[j]*sin( (float)j * omega );
//      printf("%i: %f vs %f\n", j, bvh, b_h[j]);
      c_dh[i] += b_h[j];
      c_hh[i] += bvh;
    }
  }


  for (i=0; i<M; ++i) printf("%i: %f\t/\t%f\n", i, c_hh[i], c_dh[i]);

  free(a_h); free(b_h); free(c_dh); free(c_hh); cudaFree(a_d); cudaFree(b_d);

  return 0;
}
