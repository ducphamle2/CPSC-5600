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

    bool isLeaf(int i)
    {
        throw std::invalid_argument("is leaf error");
    }
    int left(int i)
    {
        throw std::invalid_argument("left error");
    }
    int right(int i)
    {
        throw std::invalid_argument("right error");
    }
};

string
tf(bool cond)
{
    return cond ? "true" : "false";
}

void randomTest(int size, int range)
{
    Heap heap1;
    Heap heap2;

    // add a bunch of things
    cout << "Empty heap1: " << tf(heap1.empty()) << endl;
    cout << "Empty heap2: " << tf(heap2.empty()) << endl;
    for (int i = 0; i < size; i++)
    {
        int n = rand() % range;
        heap1.enqueue(n);
        heap2.enqueue(n);
        if (i % 4 == 0)
            heap1.dequeue(); // so we can get some dequeues into the mix
    }
    cout << "Filled 1: " << tf(!heap1.empty()) << endl;
    cout << "Heap1 valid: " << tf(heap1.isValid()) << endl;
    cout << "Filled 2: " << tf(!heap2.empty()) << endl;
    cout << "Heap2 valid: " << tf(heap2.isValid()) << endl;
}

void drain(Heap heap)
{
    // take them out (and check peek at the same time)
    int prev = -1;
    while (!heap.empty())
    {
        if (heap.peek() < prev)
        {
            cout << "out of order FAIL!!" << endl;
            return;
        }
        prev = heap.peek();
        if (prev != heap.dequeue())
        {
            cout << "peek != dequeue FAIL!!" << endl;
            return;
        }
    }
}

void heapifyTest(int size, int range)
{
    int data[size];
    for (int i = 0; i < size; i++)
        data[i] = rand() % range;
    Heap heap(data, size);
    cout << "Heapify test: " << (heap.isValid() ? "valid" : "INVALID") << endl;
    drain(heap);
}

void heapsortTest(int size, int range, bool print)
{
    int data[size];
    for (int i = 0; i < size; i++)
        data[i] = rand() % range;
    Heap::heapsort(data, size);
    if (print)
    {
        cout << "sorted: " << endl;
        for (int i = 0; i < size; i++)
            cout << data[i] << " ";
        cout << endl;
    }
}

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
