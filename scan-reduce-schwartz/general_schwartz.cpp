/**
 * @file general_schwartz.cpp - example of using GeneralScanSchwartz class, in this case for max
 * @author Kevin Lundeen
 * @version 3-Feb-2020
 *
 * For Seattle University, CPSC 5600, Week 5
 */

#include <iostream>
#include <chrono>
#include <random>
#include "GeneralScanSchwartz.h"

/**
 * A max reduce/scan class using GeneralScanSchwartz
 *
 * @tparam NumType  the type of object being maximized. Must support > operator.
 */
template<typename NumType>
class MaxScan : public GeneralScanSchwartz<NumType> {
public:
    MaxScan(const typename GeneralScanSchwartz<NumType>::RawData *data) : GeneralScanSchwartz<NumType>(data, 1) {
    }

protected:
    virtual NumType init() const {
        return std::numeric_limits<NumType>::min();
    }

    virtual NumType prepare(const NumType &datum) const {
        return datum;
    }

    virtual NumType combine(const NumType &left, const NumType &right) const {
        if (left > right)
            return left;
        else
            return right;
    }

    virtual NumType gen(const NumType &tally) const {
        return tally;
    }

    virtual void accum(NumType &accumulator, const NumType &right) const {
        if (accumulator < right)
            accumulator = right;
    }
};

/**
 * TallyType for LowTen
 */
struct Ten {
    int ten[10];
};

std::ostream &operator<<(std::ostream &out, const Ten &ten) {
    out << "[";
    for (int i = 0; i < 9; i++)
        out << ten.ten[i] << ", ";
    out << ten.ten[9] << "]";
    return out;
}

/**
 * Result is ten lowest numbers seen so far.
 */
class LowTen : public GeneralScanSchwartz<int, Ten> {
public:
    LowTen(const std::vector<int> *data) : GeneralScanSchwartz<int, Ten>(data) {
    }

protected:
    virtual Ten init() const {
        Ten t;
        for (int i = 0; i < 10; i++)
            t.ten[i] = std::numeric_limits<int>::max();
        return t;
    }

    virtual Ten prepare(const int &datum) const {
        Ten t;
        for (int i = 1; i < 10; i++)
            t.ten[i] = std::numeric_limits<int>::max();
        t.ten[0] = datum;
        return t;
    }

    /**
     * A little mini-merge of the two sides.
     * @param left  lowest ten of some set
     * @param right lowest ten of another set
     * @return      lowest ten of combined set
     */
    virtual Ten combine(const Ten &left, const Ten &right) const {
        Ten t;
        int r = 0, l = 0;
        for (int i = 0; i < 10; i++)
            if (left.ten[l] < right.ten[r])
                t.ten[i] = left.ten[l++];
            else
                t.ten[i] = right.ten[r++];
        return t;
    }

    virtual Ten gen(const Ten &tally) const {
        return tally;
    }

};

/**
 * Average of ten lowest numbers seen so far.
 */
class AvgLowTen : public GeneralScanSchwartz<int, Ten, double> {
public:
    AvgLowTen(const std::vector<int> *data) : GeneralScanSchwartz<int, Ten, double>(data) {}

protected:
    virtual double gen(const Ten &tally) const {
        double total = 0.0;
        int count = 0;
        for (int elem: tally.ten)
            if (elem != std::numeric_limits<int>::max()) {
                total += elem;
                count++;
            }
        return total / count;
    }

    // the rest are the same, but not readily gotten via simple inheritance
    virtual Ten init() const {
        Ten t;
        for (int i = 0; i < 10; i++)
            t.ten[i] = std::numeric_limits<int>::max();
        return t;
    }

    virtual Ten prepare(const int &datum) const {
        Ten t;
        for (int i = 1; i < 10; i++)
            t.ten[i] = std::numeric_limits<int>::max();
        t.ten[0] = datum;
        return t;
    }

    virtual Ten combine(const Ten &left, const Ten &right) const {
        Ten t;
        int r = 0, l = 0;
        for (int i = 0; i < 10; i++)
            if (left.ten[l] < right.ten[r])
                t.ten[i] = left.ten[l++];
            else
                t.ten[i] = right.ten[r++];
        return t;
    }

};

/**
 * @class Histo for the HistoScan reductions -- buckets data into 10 interior ranges
 *              plus two outlier ranges
 */
struct Histo {
    static const int N = 10;
    int bucket[N + 2];
    int hi, lo;

    Histo() : hi(100), lo(0) {
        for (int i = 0; i < N + 2; i++)
            bucket[i] = 0;
    }
};

std::ostream &operator<<(std::ostream &out, const Histo &histo) {
    out << "|";
    for (int count: histo.bucket)
        out << count << "|";
    return out;
}

/**
 * Collects a histogram from data.
 */
class HistoScan : public GeneralScanSchwartz<int, Histo> {
public:
    HistoScan(const std::vector<int> *data) : GeneralScanSchwartz<int, Histo>(data) {
    }

protected:
    virtual Histo init() const {
        Histo h;
        return h;
    }

    virtual Histo prepare(const int &datum) const {
        Histo h;
        int bucket_size = (h.hi - h.lo) / h.N;
        if (datum < h.lo)
            h.bucket[0]++;
        else if (datum >= h.hi)
            h.bucket[h.N + 1]++;
        else
            h.bucket[1 + (datum - h.lo) / bucket_size]++;
        return h;
    }

    virtual Histo combine(const Histo &left, const Histo &right) const {
        Histo h;
        for (int i = 0; i < h.N + 2; i++)
            h.bucket[i] = left.bucket[i] + right.bucket[i];
        return h;
    }

    virtual Histo gen(const Histo &tally) const {
        return tally;
    }

    /**
     * EXAMPLE of accumulator that it more efficient than the combine/replace
     * (saves 0.1 ms in our example)
     * @param acc
     * @param right
     */
    virtual void accum(Histo &acc, const Histo &right) const {
        for (int i = 0; i < acc.N + 2; i++)
            acc.bucket[i] += right.bucket[i];
    }
};

/**
 * @class ExamHeap  implements the reduce/scan from Exam 1 using GeneralScanSchwartz class
 */
class ExamHeap : public GeneralScanSchwartz<double> {
public:
    ExamHeap(const std::vector<double> *data) : GeneralScanSchwartz<double>(data) {
    }

protected:
    virtual double init() const {
        return 1.0;
    }

    virtual double prepare(const double &probability_of_miss) const {
        return 1.0 - probability_of_miss;
    }

    virtual double combine(const double &left, const double &right) const {
        return left * right;
    }

    virtual double gen(const double &tally) const {
        return tally;
    }
};

/**
 * @class SumHeap  classic sum reduction
 */
class SumHeap : public GeneralScanSchwartz<int> {
public:
    SumHeap(const std::vector<int> *data) : GeneralScanSchwartz<int>(data) {
    }

protected:
    virtual int init() const {
        return 0;
    }

    virtual int prepare(const int &datum) const {
        return datum;
    }

    virtual int combine(const int &left, const int &right) const {
        return left + right;
    }

    virtual int gen(const int &tally) const {
        return tally;
    }
};

/**
 * Execute an ExamHeap example.
 * @return  if the test was successful
 */
bool test_exam1() {
    using namespace std;
    const int N = 1 << 10;  // FIXME must be power of 2 for now
    vector<double> data(N, 1e-3);  // put a small value in each element of the data array
    vector<double> prefix(N, 1.0);

    // start timer
    auto start = chrono::steady_clock::now();

    ExamHeap heap(&data);
    cout << "exam1: " << heap.getReduction() << endl;
    heap.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    double check = 1.0;
    for (double elem: prefix) {
        check *= 1.0 - 1e-3;
        if (abs(elem - check) > 1e-8) {
            cout << "FAILED RESULT at " << check << endl;
            return false;
        }
        // cout << elem << endl;
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_hw2() {
    using namespace std;
    const int N = 1 << 27;  // FIXME must be power of 2 for now
    vector<int> data(N, 1);  // put a 1 in each element of the data array
    vector<int> prefix(N, 1);
    data[0] = 100;

    // start timer
    auto start = chrono::steady_clock::now();

    SumHeap heap(&data);
    cout << "hw2: " << heap.getReduction() << endl;
    heap.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    int check = 100;
    for (int elem: prefix)
        if (elem != check++) {
            cout << "FAILED RESULT at " << check - 1 << endl;
            return false;
        }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_max_scan() {
    using namespace std;
    const int N = 1 << 27;  // FIXME must be power of 2 for now
    vector<int> data(N, 1);  // put a 1 in each element of the data array
    vector<int> prefix(N, 1);
    data[3000] = 12345;
    data[9000] = 67890;

    // start timer
    auto start = chrono::steady_clock::now();

    MaxScan<int> max_scan(&data);
    cout << "max scan: " << max_scan.getReduction() << endl;
    max_scan.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    int check = 1;
    for (int i = 0; i < N; i++) {
        int elem = prefix[i];
        if (elem != check) {
            cout << "FAILED RESULT at " << i << ": " << elem << endl << data[i + 1] << endl;
            return false;
        } else if (i == 2999) {
            check = 12345;
        } else if (i == 8999) {
            check = 67890;
        }
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_low_ten() {
    using namespace std;
    const int N = 1 << 10;  // FIXME must be power of 2 for now
    vector<int> data(N);
    for (int i = 0; i < N; i++)
        data[i] = rand() % 100;
    vector<Ten> prefix(N);

    // start timer
    auto start = chrono::steady_clock::now();

    LowTen low_ten(&data);
    cout << "low ten: " << low_ten.getReduction() << endl;
    low_ten.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    for (int i = 0; i < 100; i++) {
        Ten elem = prefix[i];
        cout << "adding in " << data[i] << ": " << elem << endl;
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_avg_low_ten() {
    using namespace std;
    const int N = 1 << 10;  // FIXME must be power of 2 for now
    vector<int> data(N);
    for (int i = 0; i < N; i++)
        data[i] = rand() % 100;
    vector<double> prefix(N);

    // start timer
    auto start = chrono::steady_clock::now();

    AvgLowTen avg_low_ten(&data);
    cout << "avg low ten: " << avg_low_ten.getReduction() << endl;
    avg_low_ten.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    for (int i = 0; i < 100; i++) {
        double elem = prefix[i];
        cout << "adding in " << data[i] << ": " << elem << endl;
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

bool test_histo() {
    using namespace std;
    const int N = 1 << 10;  // FIXME must be power of 2 for now
    vector<int> data(N);
    for (int i = 0; i < N; i++)
        data[i] = rand() % 100;
    vector<Histo> prefix(N);

    // start timer
    auto start = chrono::steady_clock::now();

    HistoScan histo(&data);
    cout << "histo: " << histo.getReduction() << endl;
    histo.getScan(&prefix);

    // stop timer
    auto end = chrono::steady_clock::now();
    auto elpased = chrono::duration<double, milli>(end - start).count();

    for (int i = 0; i < 100; i++) {
        Histo elem = prefix[i];
        cout << "adding in " << data[i] << ": " << elem << endl;
    }
    cout << "in " << elpased << "ms" << endl;
    return true;
}

int main() {
    using namespace std;
    if (!test_histo())
        cout << "test_histo failed" << endl;
    if (!test_low_ten())
        cout << "test_low_ten failed" << endl;
    if (!test_avg_low_ten())
        cout << "test_avg_low_ten failed" << endl;
    if (!test_max_scan())
        cout << "test_max_scan failed" << endl;
    if (!test_exam1())
        cout << "test_exam1 failed" << endl;
    if (!test_hw2())
        cout << "test_hw2 failed" << endl;
    return 0;
}
