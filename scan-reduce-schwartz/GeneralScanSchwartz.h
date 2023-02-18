/**
 * @file GeneralScanSchwartz.h - tree-cap recursive version of a generic parallelized reduce/scan
 * @author Kevin Lundeen
 * @version 3-Feb-2020
 *
 * For Seattle University, CPSC 5600, Week 6
 *
 * Uses Schwartz tight-loop optimization
 */

#include <vector>
#include <future>
#include <cmath>
#include <stdexcept>

/**
 * Generalized reducing/scanning class with methods for preparing the data elements into
 * TallyType objects, combining two TallyType objects, and initializing the beginning TallyType
 * object. These methods can be overridden by subclasses to do any kind of operation for the
 * reduce/scan.
 *
 * @tparam ElemType   This is the data type of the read-only data elements.
 * @tparam TallyType  This is the combination-result data type. This type must have a 0-arg ctor.
 *                    Defaults to ElemType.
 * @tparam ResultType This is the final result data type. Any final tally will be converted to this
 *                    data type (using the gen(tally) method). Defaults to TallyType.
 */
template<typename ElemType, typename TallyType=ElemType, typename ResultType=TallyType>
class GeneralScanSchwartz {
public:
    /**
     * @class RawData - a vector of ElemType, how the raw data must be packaged for the ctor.
     */
    typedef std::vector<ElemType> RawData;

    /**
     * @class TallyData - a vector of TallyType, used for the interior nodes of the reduction.
     */
    typedef std::vector<TallyType> TallyData;

    /**
     * @class ScanData - a vector of ResultType, used for scan output.
     */
    typedef std::vector<ResultType> ScanData;

    /**
     * Interior node number of the root of the parallel reduction.
     */
    static const int ROOT = 0;

    /**
     * Default number of threads to use in the parallelization.
     */
    static const int N_THREADS = 16;  // fork a thread for top levels

    /**
     * Construct the reducer/scanner with the given input.
     * @param raw          input data
     * @param n_threads    number of threads to use for parallelization, defaults to N_THREADS
     */
    GeneralScanSchwartz(const RawData *raw, int n_threads = N_THREADS) : reduced(false), n(raw->size()), data(raw),
                                                                         height(ceil(log2(n))), n_threads(n_threads) {
        if (1 << height != n)
            throw std::invalid_argument("data size must be power of 2 for now"); // FIXME
        if (n_threads >= n)
            throw std::invalid_argument("must be more data than threads!");
        interior = new TallyData(n_threads * 2);
    }

    /**
     * Destructor removes reduction results.
     */
    virtual ~GeneralScanSchwartz() {
        delete interior;
    }

    /**
     * Get the result of the reduction at any node. The algorithm computes and saves the complete
     * reduction once and then serves subsequent requests from the stored results.
     * Node numbers are in a binary tree level ordering starting at ROOT of 0.
     * @param i    node number (defaults to ROOT)
     * @return     the reduction for the given node
     * @throws invalid_argument if the node number is invalid
     */
    ResultType getReduction(int i = ROOT) {
        if (i >= size())
            throw std::invalid_argument("non-existent node");
        reduced = reduced || reduce(ROOT); // can't do this is in ctor or virtual overrides won't work
        return gen(value(i));
    }

    /**
     * Get all the scan (inclusive) results for all the input data.
     * @param output  scan results (vector is indexed corresponding to input elements)
     */
    void getScan(ScanData *output) {
        reduced = reduced || reduce(ROOT); // need to make sure reduction has already run to get the prefix tallies
        scan(ROOT, init(), output);
    }

protected:
    /*
     * These four functions must be implemented by the subclass.
     * The fifth, accum, may be overridden by subclass to be more efficient.
     */

    /**
     * Identity element for tally operation.
     * So, combine(init(), prepare(x)) == prepare(x).
     * Typically for summing, the return is 0, for products, 1.
     * @return identity tally element
     */
    virtual TallyType init() const = 0;

    /**
     * Convert an element (in the input data) to a tally.
     * Typically, if ElemType and TallyType are the same, this just returns datum.
     * @param datum  the datum to be converted
     * @return       the corresponding tally
     */
    virtual TallyType prepare(const ElemType &datum) const = 0;

    /**
     * Combine two tallies.
     * Tallies should be commutative, i.e., combine(a,b) == combine(b,a)
     * For summing, this typically returns left + right.
     * @param left   one of the tallies to combine
     * @param right  the other of the tallies to combine
     * @return       a new tally which is the combination of left and right
     */
    virtual TallyType combine(const TallyType &left, const TallyType &right) const = 0;

    /**
     * Convert a tally to a result.
     * If the ResultType and TallyType are the same, typically this returns tally.
     * @param tally  the resultant tally to be converted
     * @return       the result indicated by the tally
     */
    virtual ResultType gen(const TallyType &tally) const = 0;

    /**
     * Combine and replace left with result.
     * (Note this is slightly different than the POPP textbook
     * which does both combine and prepare in this method.)
     * @param accumulator the tally to combine and replace
     * @param right       the other combine operand
     */
    virtual void accum(TallyType &accumulator, const TallyType &right) const {
        accumulator = combine(accumulator, right);
    }

private:
    bool reduced;  // flag to say if we've already done the initial reduction
    int n; // n is size of data, n-1 is size of interior
    const RawData *data;
    TallyData *interior;
    int height;
    int n_threads;

    /**
     * Get the value for a node in the tree.
     * If the node is in the interior, it has the required tally already.
     * If the node is a leaf, it has to get converted to a tally (via prepare).
     */
    TallyType value(int i) {
        if (i < n - 1)
            return interior->at(i);
        else
            return prepare(data->at(i - (n - 1)));
    }

    /**
     * Recursive pair-wise reduction.
     * @param i  node number
     * @return   true
     */
    bool reduce(int i) {
        if (i < n_threads - 1) {
            auto handle = std::async(std::launch::async, &GeneralScanSchwartz::reduce, this, right(i));
            reduce(left(i));
            handle.wait();
            interior->at(i) = combine(value(left(i)), value(right(i)));
        } else {
            TallyType tally = init();
            int rm = rightmost(i);
            for (int j = leftmost(i); j <= rm; j++)
                accum(tally, value(j));
            interior->at(i) = tally;
        }
        return true;
    }

    /**
     * Recursive binary-tree prefix scan (inclusive).
     * @param i           node number
     * @param tallyPrior  tally of all the elements to the left of this node
     * @param output      where to write the output results
     */
    void scan(int i, TallyType tallyPrior, ScanData *output) {
        if (i < n_threads - 1) {
            auto handle = std::async(std::launch::async, &GeneralScanSchwartz::scan, this, left(i), tallyPrior, output);
            scan(right(i), combine(tallyPrior, value(left(i))), output);
            handle.wait();
        } else {
            TallyType tally = tallyPrior;
            int rm = rightmost(i);
            for (int j = leftmost(i); j <= rm; j++) {
                accum(tally, value(j));
                output->at(j - (n - 1)) = gen(tally);
            }
        }
    }

    // Following are for maneuvering around the binary tree
    int size() {
        return (n - 1) + n;
    }

    int parent(int i) {
        return (i - 1) / 2;
    }

    int left(int i) {
        return i * 2 + 1;
    }

    int right(int i) {
        return left(i) + 1;
    }

    bool isLeaf(int i) {
        return left(i) >= size();
    }

    int leftmost(int i) {
        while (!isLeaf(i))
            i = left(i);
        return i;
    }

    int rightmost(int i) {
        while (!isLeaf(i))
            i = right(i);
        return i;
    }
};

