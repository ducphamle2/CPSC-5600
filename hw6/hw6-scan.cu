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
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups; // or...
namespace cg = cooperative_groups;

const int N = 1 << 15;
const int MAX_BLOCK_SIZE = 1024;        // n threads
const int MAX_BLOCK_SIZE_ORIGINAL = 16; // n threads

__global__ void scan(float *data, float *local)
{
    grid_group g = this_grid();
    // __shared__ float local[10];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    // if we have extra threads that we do not use then we skip (gindex > N)
    int index = gindex;
    local[index] = data[gindex];
    // #if __CUDA_ARCH__ >= 200
    //     printf("global index %d thread id: %d - block id: %d\n", gindex, threadIdx.x, blockIdx.x);
    //     printf("global index %d thread %d block %d local[%d]: %.7f \n", gindex, threadIdx.x, blockIdx.x, index, local[index]);
    // #endif
    for (int stride = 1; stride < g.group_dim().x * blockDim.x; stride *= 2)
    {
        // #if __CUDA_ARCH__ >= 200
        //         printf("global index %d thread %d block %d stride %d\n", gindex, threadIdx.x, blockIdx.x, stride);
        // #endif

        // __syncthreads(); // cannot be inside the if-block 'cuz everyone has to call it!
        g.sync();
        float addend = 0.0;
        if (stride <= index)
        {
            int newIndex = index - stride;
            addend = local[newIndex];
            // #if __CUDA_ARCH__ >= 200
            //             printf("global index %d thread %d block %d newIndex %d local[newIndex] %.7f addend: %.7f \n", gindex, threadIdx.x, blockIdx.x, newIndex, local[newIndex], addend);

            // #endif
        }
        // __syncthreads();
        g.sync();
        local[index] += addend;
        // #if __CUDA_ARCH__ >= 200
        //         printf("global index %d thread %d block %d new local[%d]: %.7f\n", gindex, threadIdx.x, blockIdx.x, index, local[index]);

        // #endif
    }
    data[gindex] = local[index];
}

__global__ void scanOriginal(float *data)
{
    __shared__ float local[10];
    int gindex = threadIdx.x;
    int index = gindex;
    local[index] = data[gindex];
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads(); // cannot be inside the if-block 'cuz everyone has to call it!
        float addend = 0.0;
        if (stride <= index)
        {
            addend = local[index - stride];
        }
        __syncthreads();
        local[index] += addend;
    }
    data[gindex] = local[index];
}

// /**
//      * Helper for printing out bits. Converts the last four bits of the given
// number to a string of 0's and 1's.
//      * @param n number to convert to a string (only last four bits are observed)
//      * @return four-character string of 0's and 1's
//      */
string fourbits(int n)
{
    string ret = /*to_string(n) + */ (n > 15 ? "/1" : "/");
    for (int bit = 3; bit >= 0; bit--)
        ret += (n & 1 << bit) ? "1" : "0";
    return ret;
}

void sort(float *x, float *y, int n)
{
    // cout << "k\tj\ti\ti^j\ti&k" << endl;
    // k is size of the pieces, starting at pairs and doubling up until we get to the whole array
    // k also determines if we want ascending or descending for each section of i's
    // corresponds to 1<<d in textbook
    for (int k = 2; k <= n; k *= 2)
    { // k is one bit, marching to the left
        // cout << fourbits(k) << "\t";
        // j is the distance between the first and second halves of the merge
        // corresponds to 1<<p in textbook
        for (int j = k / 2; j > 0; j /= 2)
        { // j is one bit, marching from k to the right
            // if (j != k / 2)
            // cout << "    \t";
            // cout << fourbits(j) << "\t";
            // i is the merge element
            for (int i = 0; i < n; i++)
            {
                // if (i != 0)
                // cout << "    \t    \t";
                // cout << fourbits(i) << "\t";
                int ixj = i ^ j; // xor: all the bits that are on in one and off in the other
                // cout
                // << fourbits(ixj) << "\t" << fourbits(i & k) << endl;
                // only compare if ixj is to the right of i
                if (ixj > i)
                {
                    if ((i & k) == 0 && x[i] > x[ixj])
                    {
                        std::swap<float>(x[i], x[ixj]);
                        std::swap<float>(y[i], y[ixj]);
                    }
                    if ((i & k) != 0 && x[i] < x[ixj])
                    {
                        std::swap<float>(x[i], x[ixj]);
                        std::swap<float>(y[i], y[ixj]);
                    }
                }
            }
        }
    }
}

void fillArray(float *y, float *data, int n, int sz)
{
    for (int i = 0; i < n; i++)
    {
        // TODO: uncomment this after the algorithm is correct
        // data[i] = y[i];
        data[i] = 1;
    }
    printf("\n");
    for (int i = n; i < sz; i++)
        data[i] = 0.0; // pad with 0.0's for addition
}

void printArrayFull(float *data, int n, string title)
{
    cout << title << ":";
    for (int i = 0; i < n; i++)
        cout << " " << data[i];
    cout << endl;
}

void printArray(float *data, int n, string title, int m = 5)
{
    cout << title << ":";
    for (int i = 0; i < m; i++)
        cout << " " << data[i];
    cout << " ...";
    for (int i = n - m; i < n; i++)
        cout << " " << data[i];
    cout << endl;
}

void readCsv(float *x, float *y, int n)
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
        x[i] = stof(word);
        row.push_back(word);
        getline(s, word, ',');
        y[i] = stof(word);
        row.push_back(word);
        // for (int j = 0; j < int(row.size()); j++)
        // {
        //     cout << row[j] << " ";
        // }
        // printf("\n");
    }
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int main(void)
{
    float *x = new float[N];
    float *y = new float[N];
    float *data, *local;
    readCsv(x, y, N);
    sort(x, y, N);
    printf("print x & y after sorting\n");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%.7f, %.7f\n", x[i], y[i]);
    // }
    int threads = MAX_BLOCK_SIZE;
    int numBlocks = (N + (threads - 1)) / threads; // total blocks we need
    printf("num block: %d\n", numBlocks);
    int size = threads * numBlocks;
    printf("size: %d\n", size);
    // cout << "How many data elements? ";
    // cin >> n;
    if (N > size)
    {
        cerr << "Cannot do more than " << threads << " numbers with this simple algorithm!" << endl;
        return 1;
    }
    printf("foobar\n");
    cudaMallocManaged(&data, (N + size) * sizeof(*data));
    cudaMallocManaged(&local, (N + size) * sizeof(*local));
    fillArray(y, data, N, size);
    // scan<<<numBlocks, threads>>>(data, local);
    void *kernelArgs[] = {&data, &local};
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
    /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, scan, threads, 0);
    printf("number of allowed blocks: %d\n", numBlocksPerSm);
    cudaLaunchCooperativeKernel((void *)scan, dimGrid, dimBlock, kernelArgs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();
    printArray(data, N, "Scan");

    // original scan
    // fillArray(y, data, N, MAX_BLOCK_SIZE_ORIGINAL);
    // scanOriginal<<<1, MAX_BLOCK_SIZE_ORIGINAL>>>(data);
    // cudaDeviceSynchronize();
    // printArray(data, N, "Scan original");
    return 0;
}