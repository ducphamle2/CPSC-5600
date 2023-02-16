#include <iostream>
#include <chrono>
#include <random>
#include "GeneralScan.h"

template <typename NumType>
class MaxScan : public GeneralScan<NumType>
{
public:
    MaxScan(const typename GeneralScan<NumType>::RawData *data) : GeneralScan<NumType>(data) {}

protected:
    virtual NumType init()
    {
        return 0;
    }

    virtual NumType prepare(NumType element)
    {
        return element;
    }

    virtual NumType combine(NumType left, NumType right)
    {
        if (left > right)
            return left;
        return right;
    }
};

class DoubleMaxScan : public MaxScan<double>
{
public:
    DoubleMaxScan(const GeneralScan<double>::RawData *data) : MaxScan<double>(data) {}
};

struct Ten
{
    double ten[10];
};

class TenScan : public GeneralScan<double, Ten>
{

public:
    TenScan(const GeneralScan<double>::RawData *data) : GeneralScan<double, Ten>(data)
    {
    }

    Ten init()
    {
        Ten t;
        for (int i = 0; i < 10; i++)
        {
            t.ten[i] = 10000; // this make sures we will collect 10 smallest values because our input will surely be smaller than them
        }
        return t;
    }

    Ten prepare(double element)

    {
        Ten t;
        for (int i = 0; i < 10; i++)
        {
            t.ten[i] = 10000;
        }
        t.ten[0] = element; // what for? May be because of abstraction, we cannot return double => we implicitly assign element 0 for leaf node
        return t;
    }

    Ten combine(Ten left, Ten right)
    {
        Ten t;
        int r = 0, l = 0;
        for (int i = 0; i < 10; i++)
        {
            if (left.ten[l] < right.ten[r])
            {
                t.ten[i] = left.ten[l++];
            }
            else
            {
                t.ten[i] = right.ten[r++];
            }
        }
        return t;
    }
};

class ExamHeap : GeneralScan<double>
{
public:
    ExamHeap(const GeneralScan<double>::RawData *data) : GeneralScan<double>(data) {}
    double init()
    {
        return 1.0;
    }
    double prepare(double datum)
    {
        return 1.0 - datum;
    }
    double combine(double left, double right)
    {
        return left * right;
    }
};

bool test_max_scan()
{
    using namespace std;
    const int N = 1 << 4; // FIXME must be power of 2 for now
    vector<double> data(N, 1);
    vector<double> prefix(N, 1);
    int max = 0;
    for (int i = 0; i < N; i++)
    {
        std::random_device rd;                           // obtain a random number from hardware
        std::mt19937 gen(rd());                          // seed the generator
        std::uniform_int_distribution<> distr(0, 10000); // define the range
        data[i] = distr(gen);
        if (data[i] > max)
        {
            max = data[i];
        };
    };

    // start timer
    auto start = chrono::steady_clock::now();

    DoubleMaxScan heap(&data);
    int reduction = heap.getReduction();
    if (reduction != max)
    {
        cout << "Invalid reduction value. Wanted: " << max << " Got: " << reduction << endl;
        return false;
    }
    heap.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    // int tally = 0;
    // for (int element : prefix)
    // {
    //     cout << "element: " << element << endl;
    // };

    // check data
    max = 0;
    for (int check = 0; check < N; check++)
    {
        int element = data[check];
        if (element > max)
        {
            max = element;
        };
        int prefix_max = prefix[check];
        if (max != prefix_max)
        {
            cout << "FAILED RESULT at " << check << endl;
            return false;
        }
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_ten()
{
    using namespace std;
    const int N = 1 << 4; // FIXME must be power of 2 for now
    vector<double> data(N, 1);
    vector<double> prefix(N, 1);
    for (int i = 0; i < N; i++)
    {
        std::random_device rd;                        // obtain a random number from hardware
        std::mt19937 gen(rd());                       // seed the generator
        std::uniform_int_distribution<> distr(0, 20); // define the range
        data[i] = distr(gen);
    };

    cout << "initial data" << endl;
    for (double element : data)
    {
        cout << "element initially: " << element << endl;
    }

    TenScan heap(&data);
    Ten reduction = heap.getReduction();
    for (double element : reduction.ten)
    {
        cout << "element: " << element << endl;
    }
    return true;
    // if (reduction != max)
    // {
    //     cout << "Invalid reduction value. Wanted: " << max << " Got: " << reduction << endl;
    //     return false;
    // }
    // heap.getScan(&prefix);
}

int main()
{
    using namespace std;
    // bool result = test_max_scan();
    // if (!result)
    // {
    //     cout << "test failed" << endl;
    //     exit(1);
    // }

    bool result = test_ten();
    if (!result)
    {
        cout << "test failed" << endl;
        exit(1);
    }
    return 0;
}