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

const int LEVELS = 4; // Number of threads used to encode and decode data elements

/**
 * @class Heaper - Heap data structure with size 2 *n - 1 to store interior nodes & initial data array in one big array. Size must be power of 2.
 * Heaper implements basic traversal methods to traverse the tree and verify if a node is a leaf or not.
 */
class Heaper
{
public:
    /**
     * @brief Heaper - constructor of the Heaper class, take a vector of integers aka original data array as input
     * Assign to instance variables. n is length of the original data array
     * @param data - pointer to the original data array which is a vector of int
     */
    Heaper(const Data *data) : n(data->size()), data(data)
    {
        interior = new Data(n - 1, 0);
    }

    /**
     * @brief ~Heaper() - destructor of the Heaper class
     * This method is used to free the interior nodes' memory
     */
    virtual ~Heaper()
    {
        delete interior;
    }

    /**
     * @brief (int i) - checks if a node index is a leaf or an interior node.
     * @param i - node index
     * @return bool - if it is an interior node then return false, else return true
     */
    bool isLeaf(int i)
    {
        if (i < n - 1)
            return false;
        else
            return true;
    }

    /**
     * @brief parent - returns the parent node index given an index of a node
     * @param childIndex - the index of a child node
     * @return int - the index of that child's parent node
     */
    int parent(int childIndex)
    {
        return (childIndex - 1) / 2;
    }

    /**
     * @brief left - return the left child node given an index of a parent index node
     * @param parentIndex - index of the parent node
     * @return int - index of its left child's node
     */
    int left(int parentIndex)
    {
        return parentIndex * 2 + 1;
    }

    /**
     * @brief right - return the right child node given an index of a parent index node
     * @param parentIndex - index of the parent node
     * @return int - index of its right child's node
     */
    int right(int parentIndex)
    {
        return left(parentIndex) + 1;
    }

protected:
    int n;            // n is size of data, n-1 is size of interior
    const Data *data; // original data vector
    Data *interior;   // interior nodes, which will be built using calSum

    /**
     * @brief size of the heaper
     * (n - 1) is the total interior nodes, and n is the total vector elements.
     * The heap stores the interiors starting from index 0, and after n - 1 elements then store original data vector
     * @return int - size of the heaper array
     */
    virtual int size()
    {
        return (n - 1) + n;
    }

    /**
     * @brief value(int i) - get the value of a heap node
     * @param i - index of the heap node element.
     * @return int - value of the node
     */
    virtual int value(int i)
    {
        if (i < n - 1) // if i from 0 to n - 1 then its interior node. Otherwise it's leaf nodes aka original data vector
            return interior->at(i);
        else
            return data->at(i - (n - 1));
    }
};

/**
 * @class SumHeap - sub class of Heaper, which implements basic methods to calculate pair wise sums and prefix sums of a data vector
 */
class SumHeap : public Heaper
{
public:
    /**
     * @brief SumHeap - constructor of the SumHeap class, take a vector of integers aka original data array as input
     * Call calSum method to calculate pair wise sum.
     */
    SumHeap(const Data *data) : Heaper(data)
    {
        threadCount = 1; // initialize and reset thread count to 1 as main thread
        calcSum(0, 0);
    }

    /**
     * @brief prefixSums - calculate the prefix sum of the data vector, and put the results in a separate vector.
     */
    void prefixSums(Data *data)
    {
        prefix = data;
        threadCount = 1; // initialize and reset thread count to 1 as main thread
        calPrefix(0, 0, 0);
        // throw std::invalid_argument("prefix sums error");
    }

    Data *prefix;    // prefix pointer pointing to the vector storing all prefix sum results
    int threadCount; // total threads used in two-pass algorithms

private:
    /**
     * @brief count - increment total thread count using mutex
     */
    void count()
    {
        mtx.lock();
        // lock to count incrementally
        threadCount++;
        mtx.unlock();
    }

    /**
     * @brief calSum - calculate pair wise sum aka interior nodes of the Heap
     * @param i - node index
     * @param level - tree level starting from the root. This param is used to control how many threads we should fork.
     */
    void calcSum(int i, int level)
    {
        // if index is leaf then we dont need to sum it => do not need to do anything
        if (isLeaf(i))
        {
            return;
        }

        // according to the hw's requirement, we only need to fork 16 threads to calculate pair-wise sum
        // based on the heap with power of 2, we have a complete binary tree, each sum can be considered a thread => top 4 levels starting from the root should be 15 threads
        // 15 + 1 main thread = total 16 threads
        // if the level is larger than 4, we stop parallelizing and start doing it sequentially
        if (level < LEVELS)
        {
            /// We fork the left side of the tree, and let it run in parallel with the right side
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

    /**
     * @brief calPrefix - calculate pair wise sum aka interior nodes of the Heap
     * @param i - node index
     * @param sumPrior - the prefix sum prior to the current node.
     * @param level - tree level starting from the root. This param is used to control how many threads we should fork.
     */
    void calPrefix(int i, int sumPrior, int level)
    {
        // if the node index is leaf, then we have reached the end of the tree. We can calculate the prefix sum for that node and store in the prefix vector
        if (isLeaf(i))
        {
            /// we will traverse through the whole heap. If its leaf then original array index is i - (n - 1) aka minus interior index
            /// size of heap is n - 1 + n, and leaf starts at n - 1, so prefix at 0 is leaf at n - 1 and so on
            prefix->at(i - (n - 1)) = sumPrior + value(i);
        }
        else
        {
            /// according to the hw's requirement, we only need to fork 16 threads to calculate pair-wise sum
            /// based on the heap with power of 2, we have a complete binary tree, each sum can be considered a thread => top 4 levels starting from the root should be 15 threads
            /// 15 + 1 main thread = total 16 threads
            /// if the level is larger than 4, we stop parallelizing and start doing it sequentially
            if (level < LEVELS)
            {
                auto handle = async(launch::async, &SumHeap::calPrefix, this, left(i), sumPrior, level + 1);
                calPrefix(right(i), sumPrior + value(left(i)), level + 1);
                handle.wait();
                count();
            }
            else
            {
                calPrefix(left(i), sumPrior, level);
                calPrefix(right(i), sumPrior + value(left(i)), level);
            }
        }
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

    auto elpased = chrono::duration<double, milli>(end - start).count();

    cout << "total thread prefix sum: " << heap.threadCount << endl;

    int check = 10;
    // loop the check failed result
    for (int elem : *heap.prefix)
        if (elem != check++)
        {
            cout << "FAILED RESULT at " << check - 1;
            break;
        }
    cout << "in " << elpased << "ms" << endl;
    return 0;
}
