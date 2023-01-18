//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#include <iostream>
#include "Heap.h"
#include <vector>
#include <chrono>
using namespace std;

typedef std::vector<int> Data; // define new vector type int called data

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
        calcSum(0);
    }

    int sum(int node = 0)
    {
        return value(node);
    }

    void prefixSums(Data *data)
    {
        throw std::invalid_argument("prefix sums error");
    }

private:
    bool isLeaf(int i)
    {
        if (i < n - 1)
            return false;
        else
            return true;
    }

    void calcSum(int i)
    {
        if (isLeaf(i))
        {
            return;
        }
        calcSum(left(i));
        calcSum(right(i));
        interior->at(i) = value(left(i)) + value(right(i));
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

int main()
{
    Data data(N, 1); // put a 1 in each element of the data array
    data[0] = 10;
    Data prefix(N, 1);

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);
    heap.prefixSums(&prefix);

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
