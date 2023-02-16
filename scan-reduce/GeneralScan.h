
#include <vector>
#include <future>
#include <cmath>
#include <stdexcept>

template <typename ElemType, typename TallyType = ElemType>
class GeneralScan
{
public:
    typedef std::vector<ElemType> RawData;
    typedef std::vector<TallyType> TallyData;

    const int N_THREADS = 16; // fork a thread for top levels

    GeneralScan(const RawData *raw) : reduced(false), n(raw->size()), data(raw), height(ceil(log2(n)))
    {
        if (1 << height != n)
            throw std::invalid_argument("data size must be power of 2 for now"); // FIXME
        interior = new TallyData(n - 1);
    }

    TallyType getReduction(int i = 0)
    {
        reduced = reduced || reduce(ROOT); // can't do this is in ctor or virtual overrides won't work
        return value(i);
    }

    void getScan(TallyData *output)
    {
        reduced = reduced || reduce(ROOT); // need to make sure reduction has already run to get the prefix tallies
        scan(ROOT, init(), output);
    }

protected:
    /*
     * These three functions are to be overridden by the subclass
     */

    /**
     * The init method is called once at the beginning of a scan to have the
     * seed element for the prior value of the first element.
     *
     * For example in a GeneralScan<double> implementing a +scan:
     *      return 0.0;
     *
     * @return a "blank" TallyType
     */
    virtual TallyType init() = 0;

    /**
     * The prepare method is called to convert a single element from the data
     * array into a TallyType of that one element.
     *
     * For example in a GeneralScan<double> implementing a +scan:
     *      return datum;
     *
     * @param datum the data element to convert
     * @return a one-element tally encompassing the one datum
     */
    virtual TallyType prepare(ElemType datum) = 0;

    /**
     * The combine element takes two tallies and returns their combined value.
     *
     * For example in a GeneralScan<double> implementing a +scan:
     *      return left + right;
     *
     * @param left   one tally, e.g., tally of entire left subtree
     * @param right  another tally, e.g., tally of entire right subtree
     * @return tally which is the combination of left and right tallies
     */
    virtual TallyType combine(TallyType left, TallyType right) = 0;

private:
    const int ROOT = 0;
    bool reduced;
    int n; // n is size of data, n-1 is size of interior
    const RawData *data;
    TallyData *interior;
    int height;

    int size()
    {
        return (n - 1) + n;
    }
    TallyType value(int i)
    {
        if (i < n - 1)
            return interior->at(i);
        else
            return prepare(data->at(i - (n - 1)));
    }
    int parent(int i)
    {
        return (i - 1) / 2;
    }
    int left(int i)
    {
        return i * 2 + 1;
    }
    int right(int i)
    {
        return left(i) + 1;
    }
    bool isLeaf(int i)
    {
        return right(i) >= size();
    }

    bool reduce(int i)
    {
        if (!isLeaf(i))
        {
            if (i < N_THREADS - 2)
            {
                auto handle = std::async(std::launch::async, &GeneralScan::reduce, this, left(i));
                reduce(right(i));
                handle.wait();
            }
            else
            {
                reduce(left(i));
                reduce(right(i));
            }
            interior->at(i) = combine(value(left(i)), value(right(i)));
        }
        return true;
    }

    void scan(int i, TallyType tallyPrior, TallyData *output)
    {
        if (isLeaf(i))
        {
            output->at(i - (n - 1)) = combine(tallyPrior, value(i));
        }
        else
        {
            if (i < N_THREADS - 2)
            {
                auto handle = std::async(std::launch::async, &GeneralScan::scan,
                                         this, left(i), tallyPrior, output);
                scan(right(i), combine(tallyPrior, value(left(i))), output);
                handle.wait();
            }
            else
            {
                scan(left(i), tallyPrior, output);
                scan(right(i), combine(tallyPrior, value(left(i))), output);
            }
        }
    }
};
