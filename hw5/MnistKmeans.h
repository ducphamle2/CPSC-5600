/**
 * @file MnistKmeans.h - a subclass of KMeansMPI to cluster Mnist images
 * @author Le Duc Pham
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#pragma once
#include "KMeansMPI.h"
#include "Mnist.h"

template <int k>
class MnistKmeans : public KMeansMPI<k, 784>
{
public:
    void fit(Mnist *images, int n)
    {
        // We know that a Mnist image is actually just an array of 784 bytes so the cast is ok
        // NOTE: this will stop working correctly if the Mnist data layout is changed in any way
        KMeansMPI<k, 784>::fit(reinterpret_cast<std::array<u_char, 784> *>(images), n);
    }

protected:
    typedef std::array<u_char, 784> Element;
    /**
     * We supply the distance method to the abstract KMeansMPI class
     * We use the Euclidean distance between the colors interpreted as 784-d vectors
     * @param a one image
     * @param b and another
     * @return distance between a and b
     */
    double distance(const Element &a, const Element &b) const override
    {
        return Mnist(a).euclidDistance(Mnist(b));
    }
};
