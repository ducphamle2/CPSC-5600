/**
 * Kevin Lundeen, Seattle University, CPSC 5600
 * bitonic_naive.cu - a bitonic sort that only works when the j-loop fits in a single block
 *                  - n must be a power of 2
 */
#include <iostream>
#include <random>
using namespace std;

const int MAX_BLOCK_SIZE = 1024; // true for all CUDA architectures so far
const int N = 1 << 20;

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
__global__ void bitonic(float *data, int k, int j, int blockId, int size)
{
    int i = blockDim.x * blockId + threadIdx.x;
    // skip threads that are out of bound for the data
    if (i >= size)
        return;
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

void fillArray(float *data, int n, int sz)
{
    int count = n;
    for (int i = 0; i < n; i++)
    {
        data[i] = count--;
    }
    for (int i = n; i < sz; i++)
        data[i] = std::numeric_limits<int>::max(); // pad with maximum for addition
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

void printArrayFull(float *data, int n, string title)
{
    cout << title << ":";
    for (int i = 0; i < n; i++)
        printf(" %.0lf", data[i]);
    cout << endl;
}

int calculateSize()
{
    // if its already a power of 2 size => we do nothing
    double result = log2(N);
    double intpart;

    if (modf(result, &intpart) == 0.0)
    {
        return N;
    }
    // ceil the result to get the closest power value
    int ceilPower = ceil(result);
    printf("ceil: %d\n", ceilPower);
    printf("result: %.2f\n", result);
    int numberOfPaddings = pow(2, ceilPower) - N;
    printf("number of paddings: %d\n", numberOfPaddings);
    return N + numberOfPaddings;
}

void sort(float *data, int size, int numBlocks)
{
    // sort it with naive bitonic sort
    for (int k = 2; k <= size; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)

        {
            for (int blockId = 0; blockId < numBlocks; blockId++)

            {
                // coming back to the host between values of k acts as a barrier
                // note that in later hardware (compute capabilty >= 7.0), there is a cuda::barrier avaliable
                bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k, j, blockId, size);
            }
            cudaDeviceSynchronize();
        }
    }
}

int main()
{
    int size = calculateSize();
    printf("size: %d\n", size);
    int threads = MAX_BLOCK_SIZE;
    int numBlocks = (size + (threads - 1)) / threads; // total blocks we need
    printf("num block: %d\n", numBlocks);

    // use managed memory for the data array
    float *data, *result;
    cudaMallocManaged(&result, N * sizeof(*data));
    cudaMallocManaged(&data, size * sizeof(*data));
    fillArray(data, N, size);

    if (size > 5)
        printArray(data, size);
    else
        printArrayFull(data, size, "before sorting array");

    // print out results
    sort(data, size, numBlocks);
    memcpy(result, data, N * sizeof(*data));
    if (N > 5)
        printArray(result, N);
    else
        printArrayFull(result, N, "after sorting array");

    cudaFree(data);
    return 0;
}