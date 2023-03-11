/**
 * hw6.cu - using bitonic sort for sorting the csv file and scanning a list of floating numbers with CUDA
 * Le Duc Pham, Seattle University, CPSC 5600
 * Notes:
 * Only works for N <= 1 << 20
 * Works with any number of lines and any number of threads in a block
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

const int N = 1 << 20;           // maximum number of lines we can process
const int MAX_BLOCK_SIZE = 1024; // n threads in a block

/**
 * The data structure of the csv file. Each line has an 'x' and 'y' value. 'originalIndex' keeps track of the index of each line before sorting
 */
struct Data
{
    float x;
    float y;
    int originalIndex;
};

// sort functions
/**
 * swaps the given elements in the given array
 * (note the __device__ moniker that says it can only
 *  be called from other device code; we don't need it
 *  here, but __device__ functions can return a value
 *  even though __global__'s cannot)
 * @param data the data that we want two elements to be swapped
 * @param a first index that will be swapped
 * @param b second index that will be swapped
 */
__device__ void swap(Data *data, int a, int b)
{

    Data temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

/**
 * Inside of the bitonic sort loop for a particular value of i for a given value of k and j
 * Because each j must be handled sequentially, we can move j out to the host
 * @param data the data pointer that we want to be sorted
 * @param k level k of the bitonic loop
 * @param j inner level j of the bitonic loop
 * @param blockId the current block thread we are handling. This is to mimic the blockIdx.x that CUDA has
 * @param size size of the data pointer
 */
__global__ void bitonic(Data *data, int k, int j, int blockId, int size)
{
    int i = blockDim.x * blockId + threadIdx.x;
    // skip threads that are out of bound for the data
    if (i >= size)
        return;
    int ixj = i ^ j;
    // avoid data race by only having the lesser of ixj and i actually do the comparison
    if (ixj > i)
    {
        if ((i & k) == 0 && data[i].x > data[ixj].x)
            swap(data, i, ixj);
        if ((i & k) != 0 && data[i].x < data[ixj].x)
            swap(data, i, ixj);
    }
}

/**
 * Inside of the bitonic sort loop for a particular value of i for a given value of k
 * @param data the data pointer that we want to be sorted
 * @param k level k of the bitonic loop
 * @param size size of the data pointer
 */
__global__ void bitonicSmall(Data *data, int k, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // skip threads that are out of bound for the data
    if (i >= size)
        return;
    for (int j = k / 2; j > 0; j /= 2)
    {
        int ixj = i ^ j;
        // avoid data race by only having the lesser of ixj and i actually do the comparison
        if (ixj > i)
        {
            if ((i & k) == 0 && data[i].x > data[ixj].x)
                swap(data, i, ixj);
            if ((i & k) != 0 && data[i].x < data[ixj].x)
                swap(data, i, ixj);
        }
        // wait for all the threads to finish before the next comparison/swap
        __syncthreads();
    }
}

/**
 * This is a wrapper for the device's bitonic loop function which puts j in to the device function when the size is small.
 * @param data the data pointer that we want to be sorted
 * @param size size of the data pointer
 * @param numBlocks number of blocks needed to sort
 */
void sortSmall(Data *data, int size, int numBlocks)
{
    // sort it with naive bitonic sort
    for (int k = 2; k <= size; k *= 2)
    {
        bitonicSmall<<<1, MAX_BLOCK_SIZE>>>(data, k, size);
    }
    cudaDeviceSynchronize();
}

/**
 * This is a wrapper for the device's bitonic loop function. It performs the bitonic loop except for the most inner loop i, which is handled by the GPU device
 * @param data the data pointer that we want to be sorted
 * @param size size of the data pointer
 * @param numBlocks number of blocks needed to sort
 */
void sort(Data *data, int size, int numBlocks)
{
    // if size is small, we move j into the device function
    if (size <= MAX_BLOCK_SIZE)
    {
        cout << "Call bitonic sort small" << endl;
        sortSmall(data, size, numBlocks);
        return;
    }
    cout << "Call bitonic sort large" << endl;

    // sort it with naive bitonic sort
    for (int k = 2; k <= size; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)

        {
            for (int blockId = 0; blockId < numBlocks; blockId++)
            {
                // coming back to the host between values of j acts as a barrier
                // note that in later hardware (compute capabilty >= 7.0), there is a cuda::barrier avaliable
                bitonic<<<1, MAX_BLOCK_SIZE>>>(data, k, j, blockId, size);
            }
            cudaDeviceSynchronize();
        }
    }
}

// scan functions
/**
 * Scan a block chunk given a block id
 * @param data data pointer that we want to scan
 * @param blockId the block that we are currently handling
 * @param size data size
 * @param sums the 2nd tier sum array. This array is used to accumulate the sum of each chunk for the next one.
 */
__global__ void scan(Data *data, int blockId, int size, float *sums)
{
    __shared__ float local[MAX_BLOCK_SIZE];
    int gindex = threadIdx.x + blockId * blockDim.x;
    int index = threadIdx.x; // because we are using a shared local, local index = threadId
    if (gindex >= size)
        return;
    local[index] = data[gindex].y;
    // printf("thread %d local[%d]: %.15lf \n", gindex, index, local[index]);
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
        // printf("thread %d source %d addend: %.15lf \n", gindex, index - stride, addend);
        // printf("thread %d new local[%d]: %.15lf\n", gindex, index, local[index]);
    }
    // final index of the chunk which is the reduction of the chunk. We collect its sum to put in the 2nd tier
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
}

/**
 * host Scan wrapper function of the device one. It loops through the number of blocks and call the device scan for each chunk
 * @param data data pointer that we want to scan
 * @param size data size
 * @param numBlocks number of blocks needed to scan
 */
void handleScan(Data *data, int size, int numBlocks)
{
    float *sums;
    cudaMallocManaged(&sums, numBlocks * sizeof(*sums));
    for (int i = 0; i < numBlocks; i++)
    {
        scan<<<1, MAX_BLOCK_SIZE>>>(data, i, size, sums);
        cudaDeviceSynchronize();
    }
}

// utilities

/**
 * Fill the array with paddings so that the algorithms can work with any number of rows
 * @param data pointer data that we want to fill in
 * @param n the original size of the data
 * @param sz the additional size that we want to add paddings
 */
void fillArray(Data *data, int n, int sz)
{
    for (int i = n; i < sz; i++)
    {
        data[i].x = std::numeric_limits<int>::max(); // pad with maximum for sorting
        data[i].y = 0.0;                             // pad with 0 for scanning
    }
}

void printArrayShort(Data *data, int n, int m = 5)
{
    for (int i = 0; i < m; i++)
        cout << " " << data[i].x << ", " << data[i].y << endl;
    cout << " ..." << endl;
    for (int i = n - m; i < n; i++)
        cout << " " << data[i].x << ", " << data[i].y << endl;
    cout << endl;
}

void printArrayFull(Data *data, int n)
{
    for (int i = 0; i < n; i++)
        printf(" %.3lf,%.3lf\n", data[i].x, data[i].y);
    cout << endl;
}

/**
 * print the array depending on its size
 */
void printArray(Data *data, int n)
{
    if (n <= 5)
        printArrayFull(data, n);
    else
        printArrayShort(data, n);
}

/**
 * Read the total number of lines of a file
 * @param filename the name of the file we want to read
 */
int readCsvLines(const char *filename)
{
    fstream fin;
    // Open an existing file
    fin.open(filename, ios::in);
    if (fin.peek() == std::ifstream::traits_type::eof())
    {
        throw runtime_error("The file is empty!");
    }

    // Read the Data from the file
    // as String Vector
    string line, word, temp;
    int n_lines = -1; // we dont count the first row, which is x, y

    // count the number of lines in a file
    while (fin.peek() != EOF)
    {
        getline(fin, line);
        n_lines++;
    }
    fin.close();
    // we dont accept files that are larger than 1 << 20 lines
    if (n_lines > N)
    {
        throw runtime_error("The file size is larger than 1 << 20. Invalid!");
    }
    return n_lines;
}

/**
 * Read the content of a csv file given a pre-defined line count
 * @param data - data pointer we want to put the content in
 * @param n - total line count of the file
 * @param filename - filename to collect content
 */
void readCsv(Data *data, int n, const char *filename)
{
    // File pointer
    fstream fin;
    fin.open(filename, ios::in);
    if (fin.peek() == std::ifstream::traits_type::eof())
    {
        throw runtime_error("The file is empty!");
    }

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
        row.push_back(word);
    }
}

/**
 * Calculate the nearest power of 2 value given an integer
 * This function helps us calculate the number of paddings we need for the file to have the size of a power of 2
 * @param n original size
 */
int calculateSize(int n)
{
    double result = log2(n);
    double intpart;

    // if its already a power of 2 size => we return it
    if (modf(result, &intpart) == 0.0)
    {
        return n;
    }
    // ceil the result to get the closest power value
    int ceilPower = ceil(result);
    return pow(2, ceilPower);
}

/**
 * write the processed data to a csv file
 */
void writeStdout(Data *data, Data *sortedData, int n)
{
    ofstream myfile("output.csv");

    if (myfile.is_open())
    {
        myfile << "n,x,y,scan"
               << "\n";

        for (int i = 0; i < n; i++)
        {
            // we display sorted-x,sorted-y,prefix-y,original-index per line
            myfile << data[i].originalIndex + 1 << "," << sortedData[i].x << "," << sortedData[i].y << "," << data[i].y << "\n";
        }
        myfile.close();
    }
}

int main(void)
{
    string filename;
    cout << "Type the csv file you want to read: ";
    cin >> filename; // get user input from the keyboard
    const char *filename_raw = filename.c_str();
    int n = readCsvLines(filename_raw);
    Data *data, *sortedData;
    // prepare size for sorting & scanning
    int size = calculateSize(n);
    printf("size: %d\n", size);
    int numBlocks = (size + (MAX_BLOCK_SIZE - 1)) / MAX_BLOCK_SIZE; // total blocks we need
    printf("num block: %d\n", numBlocks);

    cudaMallocManaged(&data, size * sizeof(*data));
    sortedData = (Data *)malloc(n * sizeof(*sortedData));

    // retrieve the content of the file & put it into the data variable
    readCsv(data, n, filename_raw);
    printf("print x & y before sorting & scanning\n");
    printArray(data, n);

    // sort
    fillArray(data, n, size);
    sort(data, size, numBlocks);
    memcpy(sortedData, data, n * sizeof(*sortedData)); // keep a copy of the original to write to csv file

    // scan
    handleScan(data, size, numBlocks);
    printf("print x & y after scanning & sorting\n");
    printArray(data, n);

    // write to csv
    writeStdout(data, sortedData, n);
    cudaFree(data);
    free(sortedData);

    return 0;
}