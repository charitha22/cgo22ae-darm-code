#include <hip/hip_runtime.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

#define BLOCK_SIZE (1 << 4)
#define IMG_WIDTH (1 << 15)
#define IMG_HEIGHT (1 << 15)

// __constant__ short Q[] = {32, 33, 51, 81, 66, 39, 34, 17, 
//                           33, 36, 48, 47, 28, 23, 12, 12, 
//                           51, 48, 47, 28, 23, 12, 12, 12, 
//                           81, 47, 28, 23, 12, 12, 12, 12,
//                           66, 28, 23, 12, 12, 12, 12, 12, 
//                           39, 23, 12, 12, 12, 12, 12, 12, 
//                           34, 12, 12, 12, 12, 12, 12, 12, 
//                           17, 12, 12, 12, 12, 12, 12, 12};

__constant__ short Q[] = {32, 33, 51, 81, 66, 39, 34, 17, 
                          33, 36, 48, 47, 28, 23, 12, 12, 
                          51, 48, 47, 28, 23, 12, 12, 12, 
                          81, 47, 28, 23, 12, 12, 12, 12,
                          66, 28, 23, 12, 12, 12, 12, 12, 
                          39, 23, 12, 12, 12, 12, 12, 12, 
                          34, 12, 12, 12, 12, 12, 12, 12, 
                          17, 12, 12, 12, 12, 12, 12, 12,
                          32, 33, 51, 81, 66, 39, 34, 17, 
                          33, 36, 48, 47, 28, 23, 12, 12, 
                          51, 48, 47, 28, 23, 12, 12, 12, 
                          81, 47, 28, 23, 12, 12, 12, 12,
                          66, 28, 23, 12, 12, 12, 12, 12, 
                          39, 23, 12, 12, 12, 12, 12, 12, 
                          34, 12, 12, 12, 12, 12, 12, 12, 
                          17, 12, 12, 12, 12, 12, 12, 12,
                          32, 33, 51, 81, 66, 39, 34, 17, 
                          33, 36, 48, 47, 28, 23, 12, 12, 
                          51, 48, 47, 28, 23, 12, 12, 12, 
                          81, 47, 28, 23, 12, 12, 12, 12,
                          66, 28, 23, 12, 12, 12, 12, 12, 
                          39, 23, 12, 12, 12, 12, 12, 12, 
                          34, 12, 12, 12, 12, 12, 12, 12, 
                          17, 12, 12, 12, 12, 12, 12, 12,
                          32, 33, 51, 81, 66, 39, 34, 17, 
                          33, 36, 48, 47, 28, 23, 12, 12, 
                          51, 48, 47, 28, 23, 12, 12, 12, 
                          81, 47, 28, 23, 12, 12, 12, 12,
                          66, 28, 23, 12, 12, 12, 12, 12, 
                          39, 23, 12, 12, 12, 12, 12, 12, 
                          34, 12, 12, 12, 12, 12, 12, 12, 
                          17, 12, 12, 12, 12, 12, 12, 12};


__global__ void CUDAkernelQuantizationShort(short *SrcDst, int Stride) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index (current coefficient)
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // copy current coefficient to the local variable
  short curCoef =
      SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)];
  short curQuant = Q[ty * BLOCK_SIZE + tx];

  // quantize the current coefficient
  if (curCoef < 0) {

    curCoef = -curCoef;
    curCoef += curQuant >> 1;
    curCoef /= curQuant;
    curCoef = -curCoef;
  } else {

    curCoef += curQuant >> 1;
    curCoef /= curQuant;
  }

  __syncthreads();

  curCoef = curCoef * curQuant;

  // copy quantized coefficient back to the DCT-plane
  SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx)] = curCoef;
}

void init_data(short *img, int height, int width) {
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      img[i * width + j] = static_cast<short>(rand() % 255) - 128;
    }
  }
}

int main() {

  srand(0);

  short *img_d, *img_h;

  int size = IMG_WIDTH * IMG_HEIGHT;
  img_h = (short *)malloc(size * sizeof(short));

  init_data(img_h, IMG_HEIGHT, IMG_WIDTH);
  
  // for (int i = 0; i < IMG_HEIGHT; ++i) {
  //   for (int j = 0; j < IMG_WIDTH; ++j) {
  //     std::cout << img_h[i * IMG_WIDTH + j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "\n";

  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipMalloc((void **)&img_d, size * sizeof(short));
  hipMemcpy(img_d, img_h, size * sizeof(short), hipMemcpyHostToDevice);

  dim3 grid(IMG_WIDTH / BLOCK_SIZE, IMG_HEIGHT / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);

  hipEventRecord(start);
  CUDAkernelQuantizationShort<<<grid, block>>>(img_d, IMG_WIDTH);
  hipEventRecord(stop);
  hipDeviceSynchronize();

  hipMemcpy(img_h, img_d, size * sizeof(short), hipMemcpyDeviceToHost);

  hipEventSynchronize(stop);
  float milis = 0.f;
  hipEventElapsedTime(&milis, start, stop);
  std::cout << "" << milis << "\n";

  // for (int i = 0; i < IMG_HEIGHT; ++i) {
  //   for (int j = 0; j < IMG_WIDTH; ++j) {
  //     std::cout << img_h[i * IMG_WIDTH + j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  free(img_h);
  hipFree(img_d);

  hipDeviceReset();

  return 0;
}
