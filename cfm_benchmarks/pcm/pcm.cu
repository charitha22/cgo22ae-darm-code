#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
/*#include <cuda.h>*/

#define B 32
#define N 1024
#define S 4

__global__ void pcm(int *c)
{
	int l = (N % 2 == 0) ? N / 2 : (N / 2) + 1;
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int sR[N*S];
	__shared__ int sC[N*S];
	
	if (gid < B*N*S)
		for(int i = 0; i < S; i++)
			sC[tid*S+i] = c[gid*S+i];
	__syncthreads();

	int * R = (sR + tid * S);
	int * C = sC;
	for(int i = 0; i < l; i++)
	{
		//even phase
		bool cond_even = (!(tid & 1)) && (tid < (N-1));  
		if (cond_even)  
		{
			int c11 = 0; int c21 = 0; 
			for (int j = 0; j < S; j++)
			{
				if (C[tid * S + c11] <= C[(tid + 1) * S + c21]){
					R[j] = C[tid * S + c11]; c11++;
				} else {
					R[j] = C[(tid + 1) * S + c21]; c21++;
				}
			}
		} else {
			int c12 = S-1; int c22 = S-1;
			for (int j = 0; j < S; j++)
			{
				if (C[tid * S + c22] > C[(tid-1) * S + c12]) {
					R[(S - 1) - j] = C[tid * S + c22]; c22--;
				} else {
					R[(S - 1) - j] = C[(tid - 1) * S + c12]; c12--;
				}
			}
		}
		__syncthreads();
		
		if (tid < N)  
			for (int j = 0; j < S; j++) 
				C[tid * S + j] = R[j];
		__syncthreads();
		
		//odd phase
		bool cond_odd = (tid & 1) && (tid < (N-1)); 
		if (cond_odd)  
		{
			int c11 = 0; int c21 = 0; 
			for (int j = 0; j < S; j++)
			{
				if (C[tid * S + c11] <= C[(tid + 1) * S + c21]){
					R[j] = C[tid * S + c11]; c11++;
				} else {
					R[j] = C[(tid + 1) * S + c21]; c21++;
				}
			}
		} else {
			if (tid > 0) {
				int c12 = S-1; int c22 = S-1;
				for (int j = 0; j < S; j++)
				{
					if (C[tid * S + c22] > C[(tid-1) * S + c12]) {
						R[(S - 1) - j] = C[tid * S + c22]; c22--;
					} else {
						R[(S - 1) - j] = C[(tid - 1) * S + c12]; c12--;
					}
				}
			}
		}
		__syncthreads();
		
		if (tid < N)  
			for (int j = 0; j < S; j++) 
				C[tid * S + j] = R[j];
		__syncthreads();
	}//for

	if (gid < B*N*S)
		for(int i = 0; i < S; i++)
			c[gid*S+i] = sC[tid*S+i];
	__syncthreads();
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
	int a[B*N*S], b[B*N*S];
	
	// provide sorted subsequences
	for (int k = 0; k < B; k++){
		for(int i = 0; i < N; i++){ 
			int x = rand() % (B*N*S);
			for(int j = 0; j < S; j++)
				a[k*N*S+i*S+j] = x + j;
		}
	}

	// original array
	#ifdef DEBUG
	for(int k = 0; k < B; k++){
		printf("ORIGINAL ARRAY %d: \n", k);
		for(int i = 0; i < N*S; i++) 
			printf("%d ", a[k*N*S+i]);
		printf("\n");
	}
	#endif
	
	// allocate
	int *dc;
	cudaMalloc((void**)&dc, sizeof(int)*B*N*S);
	// copy
	cudaMemcpy(dc, a, sizeof(int)*B*N*S, cudaMemcpyHostToDevice);
	
	// kernel launch and copy back
	pcm<<<B, N>>>(dc);
	cudaMemcpy(b, dc, sizeof(int)*B*N*S, cudaMemcpyDeviceToHost);
	
	// sorted array
	#ifdef DEBUG
	printf("\n");
	for(int k = 0; k < B; k++){
		printf("SORTED ARRAY %d: \n", k);
		for(int i = 0; i < N * S; i++) 
			printf("%d ", b[k*N*S+i]);
		printf("\n");
	}
	#endif

	#ifdef CHECK
	printf("\n");
	for(int k = 0; k < B; k++){
		if(!check_sort(b + k*N*S, N*S))
			printf("CHECK FAIL %d\n", k);
        else
			printf("CHECK PASS %d\n", k);
	}
	#endif
	
    cudaFree(dc);
	
	return 0;
}
