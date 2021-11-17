
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
using namespace std;
// #include <cuda.h>
// #define CHECK
// #define DEBUG

#define B (1<<15)
#define N (1<<8)
#define S (1<<5)

__global__ void pcm(int *c) {
  int l = (N % 2 == 0) ? N / 2 : (N / 2) + 1;
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int sR[N * S];
  __shared__ int sC[N * S];

  if (gid < B * N * S)
    for (int i = 0; i < S; i++)
      sC[tid * S + i] = c[gid * S + i];
  __syncthreads();

  int *R = (sR + tid * S);
  int *C = sC;
  for (int i = 0; i < l; i++) {
    // even phase
    bool cond_even = (!(tid & 1)) && (tid < (N - 1));
    if (cond_even) {
      int c11 = 0;
      int c21 = 0;
      for (int j = 0; j < S; j++) {
        if (C[tid * S + c11] <= C[(tid + 1) * S + c21]) {
          R[j] = C[tid * S + c11];
          c11++;
        } else {
          R[j] = C[(tid + 1) * S + c21];
          c21++;
        }
      }
    } else {
      int c12 = S - 1;
      int c22 = S - 1;
      for (int j = 0; j < S; j++) {
        if (C[tid * S + c22] > C[(tid - 1) * S + c12]) {
          R[(S - 1) - j] = C[tid * S + c22];
          c22--;
        } else {
          R[(S - 1) - j] = C[(tid - 1) * S + c12];
          c12--;
        }
      }
    }
    __syncthreads();

    if (tid < N)
      for (int j = 0; j < S; j++)
        C[tid * S + j] = R[j];
    __syncthreads();

    // odd phase
    bool cond_odd = (tid & 1) && (tid < (N - 1));
    if (cond_odd) {
      int c11 = 0;
      int c21 = 0;
      for (int j = 0; j < S; j++) {
        if (C[tid * S + c11] <= C[(tid + 1) * S + c21]) {
          R[j] = C[tid * S + c11];
          c11++;
        } else {
          R[j] = C[(tid + 1) * S + c21];
          c21++;
        }
      }
    } else {
      if (tid > 0) {
        int c12 = S - 1;
        int c22 = S - 1;
        for (int j = 0; j < S; j++) {
          if (C[tid * S + c22] > C[(tid - 1) * S + c12]) {
            R[(S - 1) - j] = C[tid * S + c22];
            c22--;
          } else {
            R[(S - 1) - j] = C[(tid - 1) * S + c12];
            c12--;
          }
        }
      }
    }
    __syncthreads();

    if (tid < N)
      for (int j = 0; j < S; j++)
        C[tid * S + j] = R[j];
    __syncthreads();
  } // for

  if (gid < B * N * S)
    for (int i = 0; i < S; i++)
      c[gid * S + i] = sC[tid * S + i];
  __syncthreads();
}

bool check_sort(int *arr, int len) {
  for (int i = 0; i < len - 1; i++)
    if (arr[i] > arr[i + 1])
      return false;
  return true;
}

int main() {
  srand(0);
  // int a[B*N*S], b[B*N*S];
  int *a, *b;
  a = (int*)malloc(sizeof(int)*B*N*S);
  b = (int*)malloc(sizeof(int)*B*N*S);

  // provide sorted subsequences
  for (int k = 0; k < B; k++) {
    for (int i = 0; i < N; i++) {
      int x = rand() % (B * N * S);
      for (int j = 0; j < S; j++)
        a[k * N * S + i * S + j] = x + j;
    }
  }

// original array
#ifdef DEBUG
  for (int k = 0; k < B; k++) {
    printf("ORIGINAL ARRAY %d: \n", k);
    for (int i = 0; i < N * S; i++)
      printf("%d ", a[k * N * S + i]);
    printf("\n");
  }
#endif

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  // allocate
  int *dc;
  hipMalloc((void **)&dc, sizeof(int) * B * N * S);
  // copy
  hipMemcpy(dc, a, sizeof(int) * B * N * S, hipMemcpyHostToDevice);

  // kernel launch and copy back
  hipEventRecord(start);
  hipLaunchKernelGGL(pcm, dim3(B), dim3(N), 0, 0, dc);
  hipEventRecord(stop);
  hipDeviceSynchronize();
  hipMemcpy(b, dc, sizeof(int) * B * N * S, hipMemcpyDeviceToHost);

  hipEventSynchronize(stop);
  float milis = 0.f;
  hipEventElapsedTime(&milis, start, stop);
   std::cout << "" << milis << " \n";
  //printf("Time: %0.4lf ms\n", milis);
// sorted array
#ifdef DEBUG
  printf("\n");
  for (int k = 0; k < B; k++) {
    printf("SORTED ARRAY %d: \n", k);
    for (int i = 0; i < N * S; i++)
      printf("%d ", b[k * N * S + i]);
    printf("\n");
  }
#endif

#ifdef CHECK
  printf("\n");
  for (int k = 0; k < B; k++) {
    if (!check_sort(b + k * N * S, N * S))
      printf("CHECK FAIL %d\n", k);
    else
      printf("CHECK PASS %d\n", k);
  }
#endif

  hipFree(dc);
  free(a);
  free(b);
  hipDeviceReset();

  return 0;
}
