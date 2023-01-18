//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#include <iostream>
#include "Heap.h"
#include <vector>
#include <chrono>
#include <future>
using namespace std;

typedef std::vector<int> Data; // define new vector type int called data
int *vecCount = new int[100];

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
        calPrefix(data, 0, 0, 0);
        // throw std::invalid_argument("prefix sums error");
    }

private:
    bool isLeaf(int i)
    {
        if (i < n - 1)
            return false;
        else
            return true;
    }

    void calcSum(int i, int level)
    {
        if (isLeaf(i))
        {
            return;
        }

        // cout << "level: " << level << endl;

        if (level < 4)
        {
            auto leftHandle = async(launch::async, &SumHeap::calcSum, this, left(i), level + 1);
            auto rightHandle = async(launch::async, &SumHeap::calcSum, this, right(i), level + 1);
            leftHandle.wait();
            rightHandle.wait();
        }
        else
        {
            calcSum(left(i), level);
            calcSum(right(i), level);
        }
        interior->at(i) = value(left(i)) + value(right(i));
        if (level < 4)
        {
            vecCount[i]++;
        }
    }

    void calPrefix(Data *data, int i, int sumPrior, int level)
    {
        if (isLeaf(i))
        {
            // we will traverse through the whole heap. If its leaf then original array index is i - (n - 1) aka minus interior index
            // size of heap is n - 1 + n, and leaf starts at n - 1, so prefix at 0 is leaf at n - 1 and so on
            data->at(i - (n - 1)) = sumPrior + value(i);
            // cout << sumPrior + value(i) << endl;
        }
        else
        {
            if (level < 4)
            {
                auto leftHandle = async(launch::async, &SumHeap::calPrefix, this, data, left(i), sumPrior, level + 1);
                auto rightHandle = async(launch::async, &SumHeap::calPrefix, this, data, right(i), sumPrior + value(left(i)), level + 1);
                leftHandle.wait();
                rightHandle.wait();
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
};

const int N = 1 << 26; // FIXME must be power of 2 for now

Data genTestData()
{
    Data data(N, 1);
    data[0] = 2;
    data[1] = 7;
    data[2] = 1;
    data[3] = 1;
    data[4] = 4;
    data[5] = 2;
    data[6] = 9;
    data[7] = 8;
    data[8] = 7;
    data[9] = 1;
    data[10] = 4;
    data[11] = 1;
    data[12] = 2;
    data[13] = 2;
    data[14] = 3;
    data[15] = 0;
    return data;
}

int main()
{
    Data data(N, 1); // put a 1 in each element of the data array
    data[0] = 10;
    // data = genTestData();
    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);
    // heap.printSumHeap();
    heap.prefixSums(&prefix);

    // print prefix after prefix sum
    // for (int i = 0; i < N; i++)
    // {
    //     cout << "|" << prefix.at(i) << "|";
    // }
    // cout << endl;

    int total = 0;
    for (int i = 0; i < 100; i++)
    {
        if (vecCount[i] != 0)
        {
            cout << "vec count: " << vecCount[i] << endl;
            total++;
        }
    }
    cout << "total thread: " << total << endl;

    // stop timer
    auto end = chrono::steady_clock::now();
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
