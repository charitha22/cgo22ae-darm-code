#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024
#define S 512
#define swap(A, B) {int temp = A; A = B; B = temp;}

__global__ static void bitonic_sort(int *values) {
	__shared__ int shared[S];
	const unsigned int tid = threadIdx.x;
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (gid < N*S) shared[tid] = values[gid];
	__syncthreads();
  
	for (unsigned int k = 2; k <= S; k *= 2) {
	  for (unsigned int j = k / 2; j > 0; j /= 2) {
		unsigned int ixj = tid ^ j;
		if (ixj > tid) {
			if ((tid & k) == 0) {
				if (shared[tid] > shared[ixj]) swap(shared[tid], shared[ixj]);
			} else {
				if (shared[ixj] > shared[tid]) swap(shared[tid], shared[ixj]);
			}
		}
		__syncthreads();
	  }
	}
	if (gid < N*S) values[gid] = shared[tid];
}

__global__ void pcm_sort(int *c, int *r)
{
	const int l = (N % 2 == 0) ? N / 2 : (N / 2) + 1;
	const int tid = threadIdx.x;
	
	int * R = (r + tid * S);
	for(int i = 0; i < l; i++)
	{
		//even phase
		if ((!(tid & 1)) && (tid < (N-1)))  
		{
			int c11 = 0; int c21 = 0; 
			for (int j = 0; j < S; j++)
			{
				if (c[tid * S + c11] <= c[(tid + 1) * S + c21]){
					R[j] = c[tid * S + c11]; c11++;
				} else {
					R[j] = c[(tid + 1) * S + c21]; c21++;
				}
			}
		}
		else if ((tid & 1) && (tid < N))  
		{
			int c12 = S-1; int c22 = S-1;
			for (int j = 0; j < S; j++)
			{
				if (c[tid * S + c22] > c[(tid-1) * S + c12]) {
					R[(S - 1) - j] = c[tid * S + c22]; c22--;
				} else {
					R[(S - 1) - j] = c[(tid - 1) * S + c12]; c12--;
				}
			}
		}
		__syncthreads();
		
		if (tid < N)  
			for (int j = 0; j < S; j++) 
				c[tid * S + j] = R[j];
		__syncthreads();
		
		//odd phase
		if ((tid & 1) && (tid < (N-1)))  
		{
			int c11 = 0; int c21 = 0; 
			for (int j = 0; j < S; j++)
			{
				if (c[tid * S + c11] <= c[(tid + 1) * S + c21]){
					R[j] = c[tid * S + c11]; c11++;
				} else {
					R[j] = c[(tid + 1) * S + c21]; c21++;
				}
			}
		}
		else if ((!(tid & 1)) && (tid > 0) && (tid < N))  
		{
			int c12 = S-1; int c22 = S-1;
			for (int j = 0; j < S; j++)
			{
				if (c[tid * S + c22] > c[(tid-1) * S + c12]) {
					R[(S - 1) - j] = c[tid * S + c22]; c22--;
				} else {
					R[(S - 1) - j] = c[(tid - 1) * S + c12]; c12--;
				}
			}
		}
		__syncthreads();
		
		if (tid < N)  
			for (int j = 0; j < S; j++) 
				c[tid * S + j] = R[j];
		__syncthreads();
	}//for

}

bool check_sort(int * arr, int len)
{
	for (int i = 0; i < len-1; i++)
		if (arr[i] > arr[i+1]) return false;
	return true;
}

int main()
{
	srand(time(NULL));
	int a[N*S], b[N*S];
	
	// provide sorted subsequences
	for(int i = 0; i < N; i++) 
		for(int j = 0; j < S; j++)
			a[i*S+j] = rand() % (N * S);

	// original array
	#ifdef DEBUG
	printf("ORIGINAL ARRAY : \n");
	for(int i = 0; i < N * S; i++) 
		printf("%d ", a[i]);
	printf("\n");
	#endif
	
	// allocate
	int *dc, *dr;
	cudaMalloc((void**)&dc, sizeof(int)*N*S);
	cudaMalloc((void**)&dr, sizeof(int)*N*S);
	// copy
	cudaMemcpy(dc, a, sizeof(int)*N*S, cudaMemcpyHostToDevice);
	cudaMemset(dr, 0, sizeof(int)*N*S);
	
	bitonic_sort<<<N, S>>>(dc);
	pcm_sort<<<1, N>>>(dc, dr);
	
	cudaMemcpy(b, dc, sizeof(int)*N*S, cudaMemcpyDeviceToHost);
	
	// sorted array
	#ifdef DEBUG
	printf("\nSORTED ARRAY : \n");
	for(int i = 0; i < N * S; i++) 
		printf("%d ", b[i]);
	printf("\n");
	#endif

	if (check_sort(b, N*S))
		printf("\nCHECK PASS\n");
    else
		printf("\nCHECK FAIL\n");
	
	return 0;
}
