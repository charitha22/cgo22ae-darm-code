#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

// data[], size, threads, blocks,
void mergesort(long *, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long *, long *, long, long, long, dim3 *, dim3 *);
__device__ void gpu_bottomUpMerge(long *, long *, long, long, long);

#define min(a, b) (a < b ? a : b)

void init_data(long *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = static_cast<long>(rand() % 1024);
  }
}

int main(int argc, char **argv) {

  dim3 threadsPerBlock(64);
  dim3 blocksPerGrid(1024);

  int size = 1<<20;

  long *data = (long *)malloc(sizeof(long) * size);

  init_data(data, size);

  // merge-sort the data
  mergesort(data, size, threadsPerBlock, blocksPerGrid);

  free(data);

  return 0;
}

void mergesort(long *data, long size, dim3 threadsPerBlock,
               dim3 blocksPerGrid) {

  //
  // Allocate two arrays on the GPU
  // we switch back and forth between them during the sort
  //
  long *D_data;
  long *D_swp;
  dim3 *D_threads;
  dim3 *D_blocks;

  // Actually allocate the two arrays
  (hipMalloc((void **)&D_data, size * sizeof(long)));
  (hipMalloc((void **)&D_swp, size * sizeof(long)));

  // Copy from our input list into the first array
  (hipMemcpy(D_data, data, size * sizeof(long), hipMemcpyHostToDevice));

  // Copy the thread / block info to the GPU as well
  (hipMalloc((void **)&D_threads, sizeof(dim3)));
  (hipMalloc((void **)&D_blocks, sizeof(dim3)));

  (hipMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), hipMemcpyHostToDevice));
  (hipMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), hipMemcpyHostToDevice));

  long *A = D_data;
  long *B = D_swp;

  long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                  blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

  
    hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  //
  // Slice up the list and give pieces of it to each thread, letting the pieces
  // grow bigger and bigger until the whole list is sorted
  //
  hipEventRecord(start);
  for (int width = 2; width < (size << 1); width <<= 1) {
    long slices = size / ((nThreads)*width) + 1;

    gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices,
                                                      D_threads, D_blocks);

    // Switch the input / output arrays instead of copying them around
    A = A == D_data ? D_swp : D_data;
    B = B == D_data ? D_swp : D_data;
  }
  hipEventRecord(stop);
  hipDeviceSynchronize();

  (hipMemcpy(data, A, size * sizeof(long), hipMemcpyDeviceToHost));

    
  hipEventSynchronize(stop);
  float milis = 0.f;
  hipEventElapsedTime(&milis, start, stop);
  std::cout << "" << milis << "\n";


  bool pass = true;
  for (int i = 0; i < size - 1; i++) {
    if (data[i] > data[i + 1]) {
      pass = false;
      break;
    }
  }

  if (!pass) {
    std::cout << "FAIL\n";
  }
  // Free the GPU memory
  (hipFree(A));
  (hipFree(B));
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3 *threads, dim3 *blocks) {
  int x;
  return threadIdx.x + threadIdx.y * (x = threads->x) +
         threadIdx.z * (x *= threads->y) + blockIdx.x * (x *= threads->z) +
         blockIdx.y * (x *= blocks->z) + blockIdx.z * (x *= blocks->y);
}

// __device__ __noinline__ bool check_cond(int i, int j, int end, int middle, long *source ) {
//   return i < middle && (j >= end || source[i] < source[j]);
// }
//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long *source, long *dest, long size, long width,
                              long slices, dim3 *threads, dim3 *blocks) {
  unsigned int idx = getIdx(threads, blocks);
  long start = width * idx * slices, middle, end;

  for (long slice = 0; slice < slices; slice++) {
    if (start >= size)
      break;

    middle = min(start + (width >> 1), size);
    end = min(start + width, size);
    // gpu_bottomUpMerge(source, dest, start, middle, end);
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
      bool cond = i < middle && (j >= end || source[i] < source[j]);
      if (cond) {
        dest[k] = source[i];
        i++;
      } else {
        dest[k] = source[j];
        j++;
      }
    }

    start += width;
  }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
// __device__ void gpu_bottomUpMerge(long *source, long *dest, long start,
//                                   long middle, long end) {
//   long i = start;
//   long j = middle;
//   for (long k = start; k < end; k++) {
//     bool cond = i < middle && (j >= end || source[i] < source[j]);
//     if (cond) {
//       dest[k] = source[i];
//       i++;
//     } else {
//       dest[k] = source[j];
//       j++;
//     }
//   }
// }
