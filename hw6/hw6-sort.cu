/**
 * Kevin Lundeen, Seattle University, CPSC 5600
 * bitonic_naive.cu - a bitonic sort that only works when the j-loop fits in a single block
 *                  - n must be a power of 2
 */
#include <iostream>
#include <random>
using namespace std;

/**
 * swaps the given elements in the given array
 * (note the __device__ moniker that says it can only
 *  be called from other device code; we don't need it
 *  here, but __device__ functions can return a value
 *  even though __global__'s cannot)
 */
__device__ void swap(float *data, int a, int b)
{
    float temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/**
 * inside of the bitonic sort loop for a particular value of i for a given value of k
 * (this function assumes j <= MAX_BLOCK_SIZE--bigger than that and we'd need to
 *  synchronize across different blocks)
 */
__global__ void bitonic(float *data, int k, int j, int blockId)
{
    int i = blockDim.x * blockId + threadIdx.x;
    int ixj = i ^ j;
    // printf("i: %d - ixj: %d\n", i, ixj);
    // printf("i: %d\n - j: %d - k: %d\n", i, j, k);
    // printf("i & k: %d - data[%d]: %d - data[%d]: %d\n", i & k, i, data[i], ixj, data[ixj]);
    // avoid data race by only having the lesser of ixj and i actually do the comparison
    if (ixj > i)
    {
        if ((i & k) == 0 && data[i] > data[ixj])
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i] < data[ixj])
            swap(data, i, ixj);
    }
    // wait for all the threads to finish before the next comparison/swap
}

void fillArray(float *data, int n)
{
    int count = n;
    for (int i = 0; i < n; i++)
    {
        data[i] = count--;
    }
}

void printArray(float *data, int n, int m = 5)
{
    for (int i = 0; i < m; i++)
        printf(" %.0lf", data[i]);
    // cout << " " << data[i];
    cout << " ...";
    for (int i = n - m; i < n; i++)
        // cout << " " << data[i];
        printf(" %.0lf", data[i]);
    cout << endl;
}

int main()
{
    const int MAX_BLOCK_SIZE = 1024; // true for all CUDA architectures so far
    int n = 1 << 20;
    int threads = MAX_BLOCK_SIZE;
    int numBlocks = (n + (threads - 1)) / threads; // total blocks we need
    printf("num block: %d\n", numBlocks);
    int size = threads * numBlocks;

    // use managed memory for the data array
    float *data;
    cudaMallocManaged(&data, (n + size) * sizeof(*data));
    fillArray(data, n);
    printArray(data, n);

    // sort it with naive bitonic sort
    for (int k = 2; k <= n; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)

        {
            for (int blockId = 0; blockId < numBlocks; blockId++)

            {
                // coming back to the host between values of k acts as a barrier
                // note that in later hardware (compute capabilty >= 7.0), there is a cuda::barrier avaliable
                bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k, j, blockId);
            }
            cudaDeviceSynchronize();
        }
    }
    // print out results
    printArray(data, n);
    cudaFree(data);
    return 0;
}