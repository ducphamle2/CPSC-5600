/**
 * reduce_scan_1block.cu - using dissemination reduction for reducing and scanning a small array with CUDA
 * Kevin Lundeen, Seattle University, CPSC 5600 demo program
 * Notes:
 * - only works for one block (maximum block size for all of CUDA is 1024 threads per block)
 * - eliminated the remaining data races that were in reduce_scan_naive.cu
 * - algo requires power of 2 so we pad with zeros up to 1024 elements
 * - now a bit faster using block shared memory during loops (which also handily exposed the data races we had before)
 */

/**
 * Steps:
 * 1. Identify the format of the input, read & parse it
 * 2. Implement loop bitonic sort on x with 1 thread 1 block, then apply the sort to y
 * 3. Use the Scan function on y with 1 thread 1 block
 * 4. Move to scan on y with 1 block 1024 threads
 * 5. Move to bitonic sort with 1 block 1024 threads (using barrier similarly to hw4)
 * 6. Move to scan with 2^20 elements
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

const int N = 10;
const int MAX_BLOCK_SIZE = 1023; // n threads

struct Data
{
    float x;
    float y;
    int originalIndex;
};

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// sort functions
/**
 * swaps the given elements in the given array
 * (note the __device__ moniker that says it can only
 *  be called from other device code; we don't need it
 *  here, but __device__ functions can return a value
 *  even though __global__'s cannot)
 */
__device__ void swap(Data *data, int a, int b)
{

    float temp = data[a].x;
    data[a].x = data[b].x;
    data[b].x = temp;
    // also swap y
    temp = data[a].y;
    data[a].y = data[b].y;
    data[b].y = temp;

    // also swap original index
    temp = data[a].originalIndex;
    data[a].originalIndex = data[b].originalIndex;
    data[b].originalIndex = temp;
}

/**
 * inside of the bitonic sort loop for a particular value of i for a given value of k
 * (this function assumes j <= MAX_BLOCK_SIZE--bigger than that and we'd need to
 *  synchronize across different blocks)
 */
__global__ void bitonic(Data *data, int k, int j, int blockId, int size)
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
        if ((i & k) == 0 && data[i].x > data[ixj].x)
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i].x < data[ixj].x)
            swap(data, i, ixj);
    }
    // wait for all the threads to finish before the next comparison/swap
}

void sort(Data *data, int size, int numBlocks)
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

// scan functions
__global__ void scan(Data *data, int blockId, int size, float *sums)
{
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = threadIdx.x + blockId * blockDim.x;
    int index = threadIdx.x;
    if (gindex >= size)
        return;
    local[index] = data[gindex].y;
    // printf("thread %d local[%d]: %.15lf \n", gindex, index, local[index]);
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // #if __CUDA_ARCH__ >= 200
        //         printf("thread %d stride %d \n", threadIdx.x, stride);
        // #endif

        __syncthreads(); // cannot be inside the if-block 'cuz everyone has to call it!
        float addend = 0.0;
        if (stride <= index)
        {
            addend = local[index - stride];
        }
        __syncthreads();
        local[index] += addend;
        // printf("thread %d source %d addend: %.15lf \n", gindex, index - stride, addend);
        // printf("thread %d new local[%d]: %.15lf\n", gindex, index, local[index]);
    }
    // final index of the chunk, then we collect its sum to put in the 2nd tier
    if (index == MAX_BLOCK_SIZE - 1)
    {
        sums[blockId] = local[index];
        // printf("sum in index %d: %.15lf\n", gindex, sums[blockId]);
    }
    __syncthreads();
    // accumulate the sum
    for (int i = 0; i < blockId; i++)
    {
        local[index] += sums[i];
    }
    data[gindex].y = local[index];
    // printf("data[%d].y: %.15lf\n", gindex, data[gindex].y);
}

void handleScan(Data *data, int threads, int size, int numBlocks)
{
    // scan<<<numBlocks, threads>>>(data, local);
    /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    // Number of threads my_kernel will be launched with
    float *sums;
    cudaMallocManaged(&sums, numBlocks * sizeof(*sums));
    for (int i = 0; i < numBlocks; i++)
    {
        scan<<<1, MAX_BLOCK_SIZE>>>(data, i, size, sums);
        gpuErrchk(cudaPeekAtLastError());
        cudaDeviceSynchronize();
    }
}

// utilities
void fillArrayScan(float *y, float *data, int n, int sz)
{
    for (int i = 0; i < n; i++)
        data[i] = y[i];
    for (int i = n; i < sz; i++)
        data[i] = 0.0; // pad with 0.0's for addition
}

void fillArray(Data *data, int n, int sz)
{
    for (int i = n; i < sz; i++)
    {
        data[i].x = std::numeric_limits<int>::max(); // pad with maximum for addition
        data[i].y = 0.0;
    }
}

void printArrayShort(Data *data, int n, int m = 5)
{
    for (int i = 0; i < m; i++)
        printf(" %.3lf,%.3lf\n", data[i].x, data[i].y);
    // cout << " " << data[i];
    cout << " ..." << endl;
    for (int i = n - m; i < n; i++)
        // cout << " " << data[i];
        printf(" %.3lf,%.3lf\n", data[i].x, data[i].y);
    cout << endl;
}

void printArrayFull(Data *data, int n)
{
    for (int i = 0; i < n; i++)
        printf(" %.3lf,%.3lf\n", data[i].x, data[i].y);
    cout << endl;
}

void printArray(Data *data, int n)
{
    if (n <= 5)
        printArrayFull(data, n);
    else
        printArrayShort(data, n);
}

void readCsv(Data *data, int n)
{
    // File pointer
    fstream fin;
    // Open an existing file
    fin.open("x_y.csv", ios::in);

    // Read the Data from the file
    // as String Vector
    string line, word, temp;

    // skip first line with x, y
    getline(fin, line);

    for (int i = 0; i < n; i++)
    {
        vector<string> row;
        getline(fin, line);
        // used for breaking words
        stringstream s(line);
        getline(s, word, ',');
        data[i].x = stof(word);
        row.push_back(word);
        getline(s, word, ',');
        data[i].y = stof(word);
        data[i].originalIndex = i;
        // data[i].y = 0.000001;
        row.push_back(word);
    }
}

int calculateSize()
{
    // if its already a power of 2 size => we do nothing
    double result = log2(N);
    printf("result: %.2f\n", result);
    double intpart;

    if (modf(result, &intpart) == 0.0)
    {
        return N;
    }
    // ceil the result to get the closest power value
    int ceilPower = ceil(result);
    printf("ceil: %d\n", ceilPower);
    int numberOfPaddings = pow(2, ceilPower) - N;
    printf("number of paddings: %d\n", numberOfPaddings);
    return N + numberOfPaddings;
}

void writeStdout(Data *data, Data *sortedData)
{
    ofstream myfile("output.csv");

    if (myfile.is_open())
    {
        for (int i = 0; i < N; i++)
        {
            myfile << sortedData[i].x << "," << sortedData[i].y << "," << data[i].y << "," << data[i].originalIndex + 1 << "\n";
        }
        myfile.close();
    }
}

int main(void)
{
    // float value = 0.013956000097096 + 0.025529000908136 + 0.409162014722824 + 1.002696037292480 + 1.152063012123108 + 1.154309034347534 + 0.114110000431538;
    // printf("final value: %.15lf\n", value);
    Data *data, *sortedData;
    // prepare size for sorting & scanning
    int size = calculateSize();
    printf("size: %d\n", size);
    int threads = MAX_BLOCK_SIZE;
    printf("maximum block size: %d\n", threads);
    int numBlocks = (size + (threads - 1)) / threads; // total blocks we need
    printf("num block: %d\n", numBlocks);
    cudaMallocManaged(&data, size * sizeof(*data));
    sortedData = (Data *)malloc(N * sizeof(*sortedData));

    // float *data;
    readCsv(data, N);
    printf("print x & y before sorting & scanning\n");
    printArray(data, N);

    // sort
    fillArray(data, N, size);
    sort(data, size, numBlocks);
    memcpy(sortedData, data, N * sizeof(*sortedData)); // keep a copy of the original to write to csv file

    // scan
    handleScan(data, threads, size, numBlocks);
    printf("print x & y after scanning & sorting\n");
    printArray(data, N);

    // write to csv
    writeStdout(data, sortedData);
    cudaFree(data);
    free(sortedData);

    return 0;
}