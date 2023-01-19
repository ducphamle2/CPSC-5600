//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#include <iostream>
#include <vector>
#include <chrono>
#include <future>
using namespace std;

typedef std::vector<int> Data; // define new vector type int called data

const int LEVELS = 4; /// Number of threads used to encode and decode data elements

class Heaper
{
public:
    Heaper(const Data *data) : n(data->size()), data(data)
    {
        interior = new Data(n - 1, 0);
    }

    virtual ~Heaper()
    {
        delete interior;
    }

protected:
    int n; // n is size of data, n-1 is size of interior
    const Data *data;
    Data *interior;

    virtual int size()
    {
        return (n - 1) + n;
    }

    virtual int value(int i)
    {
        if (i < n - 1)
            return interior->at(i);
        else
            return data->at(i - (n - 1));
    }
};

class SumHeap : public Heaper
{
public:
    SumHeap(const Data *data) : Heaper(data)
    {
        threadCount = 1; // initialize and reset thread count to 1 as main thread
        calcSum(0, 0);
    }

    int sum(int node = 0)
    {
        return value(node);
    }

    void printSumHeap()
    {
        int i = 0;
        while (i < n - 1 + n)
        {
            cout << "|" << value(i) << "|";
            i++;
        }
        cout << endl;
    }

    void prefixSums(Data *data)
    {
        threadCount = 1; // initialize and reset thread count to 1 as main thread
        calPrefix(data, 0, 0, 0);
        // throw std::invalid_argument("prefix sums error");
    }

    int threadCount;

private:
    bool isLeaf(int i)
    {
        if (i < n - 1)
            return false;
        else
            return true;
    }

    void count()
    {
        mtx.lock();
        // lock to count incrementally
        threadCount++;
        mtx.unlock();
    }

    void calcSum(int i, int level)
    {
        if (isLeaf(i))
        {
            return;
        }

        // according to the hw's requirement, we only need to fork 16 threads to calculate pair-wise sum
        // based on the heap with power of 2, we have a complete binary tree, each sum can be considered a thread => top 4 levels starting from the root should be 15 threads
        // 15 + 1 main thread = total 16 threads
        if (level < LEVELS)
        {
            // This means that we are at the parent node's point of view, and we are creating two new threads for our left child & right child to calculate their sums.
            auto handle = async(launch::async, &SumHeap::calcSum, this, left(i), level + 1);
            calcSum(right(i), level + 1);
            handle.wait();
            count();
        }
        else
        {
            calcSum(left(i), level);
            calcSum(right(i), level);
        }
        interior->at(i) = value(left(i)) + value(right(i));
    }

    void calPrefix(Data *data, int i, int sumPrior, int level)
    {
        if (isLeaf(i))
        {
            // we will traverse through the whole heap. If its leaf then original array index is i - (n - 1) aka minus interior index
            // size of heap is n - 1 + n, and leaf starts at n - 1, so prefix at 0 is leaf at n - 1 and so on
            data->at(i - (n - 1)) = sumPrior + value(i);
        }
        else
        {
            if (level < LEVELS)
            {
                auto handle = async(launch::async, &SumHeap::calPrefix, this, data, left(i), sumPrior, level + 1);
                calPrefix(data, right(i), sumPrior + value(left(i)), level + 1);
                handle.wait();
                count();
            }
            else
            {
                calPrefix(data, left(i), sumPrior, level);
                calPrefix(data, right(i), sumPrior + value(left(i)), level);
            }
        }
    }

    int parent(int childIndex)
    {
        return (childIndex - 1) / 2;
    }

    int left(int parentIndex)
    {
        return parentIndex * 2 + 1;
    }

    int right(int parentIndex)
    {
        return left(parentIndex) + 1;
    }

    std::mutex mtx; // mutex for critical section to count total pair wise sum threads & prefix sum threads
};

const int N = 1 << 26; // FIXME must be power of 2 for now

int main()
{
    Data data(N, 1); // put a 1 in each element of the data array
    data[0] = 10;
    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);
    cout << "total thread calsum: " << heap.threadCount << endl;
    heap.prefixSums(&prefix);
    // stop timer
    auto end = chrono::steady_clock::now();

    cout << "total thread prefix sum: " << heap.threadCount << endl;

    auto elpased = chrono::duration<double, milli>(end - start).count();

    int check = 10;
    for (int elem : prefix)
        if (elem != check++)
        {
            cout << "FAILED RESULT at " << check - 1;
            break;
        }
    cout << "in " << elpased << "ms" << endl;
    return 0;
}
