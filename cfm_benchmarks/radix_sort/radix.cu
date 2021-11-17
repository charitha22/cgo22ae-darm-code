#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>
#define B 4
#define N 1024
#define UPPER_BIT 31
#define LOWER_BIT 0

__device__ void partition_by_bit(unsigned int *values, unsigned int bit);


/*******************************************************************************
   RADIX_SORT()

   For each bit position from the least significant to the most significant,
   partition the elements so that all elements with a 0 in that bit position
   precede those with a 1 in that position, using a stable sort.
   When all bits have been so processed, the array is sorted.
   Reminder -- a sort is stable if the sort preserves the relative order of 
               equal elements.

   Because this is a device function (executed by each thread concurrently),
   after each partitioning step, the threads must execute __syncthreads() so
   that the array is guaranteed to be ready for the next step.
*******************************************************************************/

__global__ void radix_sort(unsigned int *values)
{
	const unsigned int tid = threadIdx.x;
	const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned int vals[N];

	if (gid < B*N) vals[tid] = values[gid];
    __syncthreads();
    
    for(int bit = LOWER_BIT; bit <= UPPER_BIT; ++bit )
    {
        partition_by_bit(vals, bit);
        __syncthreads();
    }
	
	if (gid < B*N) values[gid] = vals[tid];
}

/*******************************************************************************

   PLUS_SCAN()

   plus_scan(a[]), where a[] is an array of integers, replaces a[] by the prefix
   sums of the elements of a. The prefix sum of an element in an array (or more 
   generally, any sequence) is the sum of all elements up to and including that
   element. The sum operation can be replaced by any binary associative operator,
   such as multiplication.

   A thread with ID i that calls plus_scan(a) gets as its return value the new 
   element in a[i]. All threads together collectively replace the elements of
   a[].
   

   Example:
                   A =  3  1  7  0  4  1  6  3 

   Successive iterations yield
       offset = 1  A =  3  4  8  7  4  5  7  9
       offset = 2  A =  3  4 11 11 12 12 11 14
       offset = 4  A =  3  4 11 11 15 16 22 25

   When it is finished it will have taken log N steps and used N log N adds.
   (This means that it is not work-efficient, since the sequential algorithm
    uses N adds.)

*******************************************************************************/
template<class T>
__device__ T plus_scan(T *x)
{
    unsigned int i = threadIdx.x; // id of thread executing this instance
    unsigned int n = blockDim.x;  // total number of threads in this block
    unsigned int offset;          // distance between elements to be added

    for( offset = 1; offset < n; offset *= 2) {
        T t;

        if ( i >= offset ) t = x[i-offset];
        __syncthreads();

        if ( i >= offset ) x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]
        __syncthreads();
    }
    return x[i];
}

/*******************************************************************************
    partition_by_bit()

    This function is executed by every thread. Given an array of non-negative
    integer values, and a bit position, b, this partitions the array such that
    for all values[i], i = 0,...,n-1, the value of bit b in each element 
    values[k]  for k < i is <= the value of bit b in values[i], and if bit b in
    values[j] == bit b in values[i], and j < i, then after the partition, the 
    two elements will be in the same relative order (i.e., it is a stable sort).

    Each thread is responsible for repositioning a single element of the array.
*******************************************************************************/

__device__ 
void partition_by_bit(unsigned int *values, unsigned int bit)
{
    unsigned int i = threadIdx.x;
    unsigned int size = blockDim.x;
    unsigned int x_i = values[i];          // value of integer at position i
    unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit

    // Replace values array so that values[i] is the value of bit bit in
    // element i.
    values[i] = p_i;  

    // Wait for all threads to finish this.
    __syncthreads();

    // Now the values array consists of 0's and 1's, such that values[i] = 0
    // if the bit at position bit in element i was 0 and 1 otherwise.

    // Compute number of True bits (1-bits) up to and including values[i], 
    // transforming values[] so that values[i] contains the sum of the 1-bits
    // from values[0] .. values[i]
    unsigned int T_before = plus_scan(values);
/*
    plus_scan(values) returns the total number of 1-bits for all j such that
    j <= i. This is assigned to T_before, the number of 1-bits before i 
    (includes i itself)
*/

    // The plus_scan() function does not return here until all threads have
    // reached the __syncthreads() call in the last iteration of its loop
    // Therefore, when it does return, we know that the entire array has had
    // the prefix sums computed, and that values[size-1] is the sum of all
    // elements in the array, which happens to be the number of 1-bits in 
    // the current bit position.
    unsigned int T_total  = values[size-1];
    // T_total, after the scan, is the total number of 1-bits in the entire array.

    unsigned int F_total  = size - T_total;
/*    
    F_total is the total size of the array less the number of 1-bits and hence
    is the number of 0-bits.
*/
    __syncthreads();

/*
    The value x_i must now be put back into the values array in the correct
    position. The array has to satisfy the condition that all values with a 0 in
    the current bit position must precede all those with a 1 in that position
    and it must be stable, meaning that if x_j and x_k both had the same bit 
    value before, and j < k, then x_j must precede x_k after sorting.

    Therefore, if x_i had a 1 in the current bit position before, it must now
    be in the position such that all x_j that had a 0 precede it, and all x_j
    that had a 1 in that bit and for which j < i, must precede it. Therefore
    if x_i had a 1, it must go into the index T_before-1 + F_total, which is the
    sum of the 0-bits and 1-bits that preceded it before (subtracting 1 since
    T_before includes x_i itself).

    If x_i has a 0 in the current bit position, then it has to be "slid" down
    in the array before all x_j such that x_j has a 1 in the current bit, but 
    no farther than that. Since there are T_before such j, it has to go to
    position i - T_before.  (There are T_before such j because x_i had a zero,
    so in the prefix sum, it does not contribute to the sum.)
*/
    if ( p_i ) values[T_before-1 + F_total] = x_i;
    else values[i - T_before] = x_i;
/*
   The interesting thing is that no two values will be placed in the same 
   position. I.e., this is a permutation of the array.

   Proof: Suppose that x_i and x_j both end up in index k. There are three
   cases: 
     Case 1. x_i and x_j have a 1 in the current bit position 
     Since F_total is the same for all threads, this implies that T_before must
     be the same for threads i and j. But this is not possible because one must 
     precede the other and therefore the one that precedes it must have smaller
     T_before.

     Case 2.  x_i and x_j both have a 0 in the current bit position. 
     Since they both are in k, we have 
         k = i - T_bef_i = j - T_Bef_j  or
         i - j = T_bef_i - T_bef_j
     Assume i > j without loss of generality.  This implies that the number of
     1-bits from position j+1 to position i-1 (since both x_j and x_i have 
     0-bits) is i-j. But that is impossible since there are only i-j-2 positions
     from j+1 to i-1.

     Case 3. x_i and x_j have different bit values. 
     Assume without loss of generality that x_j has the 0-bit and x_i, the 1-bit.
     T_before_j is the number of 1 bits in positions strictly less than j, 
     because there is a 0 in position j. The total number of positions less than
     j is j, since the array is 0-based. Therefore:

     j-T_before_j is the number of 0-bits in positions strictly less than j. 
     This must be strictly less than F_total, since x_j has a 0 in position j, 
     so there is at least one more 0 besides those below position j. Hence:

     (1)    F_total > j - T_before_j

     Turning to i, T_before_i is at least 1, since x_i has a 1 in its bit. So, 
     T_before_i - 1 is at least 0, and 

     (2)    T_before_i - 1 + F_total >= F_total. 

     Therefore, combining (1) and (2)

     (3)   T_before_i - 1 + F_total >= F_total  
                                    >  j - T_before_j

     But if x_i and x_j map to the same position, then 

     (4)   j - T_before_j  = T_before_i - 1 + F_total 
                           > j - T_before_j

     which is a contradiction since a number cannot be greater than itself!

     Therefore it is impossible for x_i and x_j to be placed in the same index
     if i != j.
     
*/
}

bool check_sort(unsigned int * arr, unsigned int len)
{
	for (unsigned int i = 0; i < len-1; i++)
		if (arr[i] > arr[i+1]) return false;
	return true;
}

int main()
{
	srand(time(NULL));
	unsigned int a[B*N], b[B*N];
	
	// provide sorted subsequences
	for(unsigned int i = 0; i < B*N; i++){ 
		a[i] = rand() % N;
	}

	// original array
	#ifdef DEBUG
	printf("\n");
	for(unsigned int k = 0; k < B; k++) {
		printf("ORIGINAL ARRAY %d: \n", k);
		for(unsigned int i = 0; i < N; i++) 
			printf("%u ", a[k*N+i]);
		printf("\n");
	}
	#endif
	
	// allocate
	unsigned int *dc;
	cudaMalloc((void**)&dc, sizeof(unsigned int)*B*N);
	// copy
	cudaMemcpy(dc, a, sizeof(unsigned int)*B*N, cudaMemcpyHostToDevice);
	
	// kernel launch and copy back
	radix_sort<<<B, N>>>(dc);
	cudaMemcpy(b, dc, sizeof(unsigned int)*B*N, cudaMemcpyDeviceToHost);
	
	// sorted array
	#ifdef DEBUG
	printf("\n");
	for(unsigned int k = 0; k < B; k++) {
		printf("SORTED ARRAY %d: \n", k);
		for(unsigned int i = 0; i < N; i++) 
			printf("%u ", b[k*N+i]);
		printf("\n");
	}
	#endif

	#ifdef CHECK
	printf("\n");
	for(unsigned int k = 0; k < B; k++) {
		if (check_sort(b+k*N, N))
			printf("CHECK PASS %u\n", k);
		else
			printf("CHECK FAIL %u\n", k);
	}
    #endif
	
    cudaFree(dc);
	cudaDeviceReset();

	return 0;
}