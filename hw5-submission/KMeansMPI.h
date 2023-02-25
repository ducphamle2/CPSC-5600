/**
 * @file KMeansMPI.h - implementation of k-means clustering algorithm using MPI
 * @author Le Duc Pham
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#pragma once // only process the first time it is included; ignore otherwise
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>
#include <cstdint>
#include <cstring>
#include "mpi.h"
using namespace std;

#define RootProcess 0

// d here is the dimensions of our data. For RGB colors it is 3
/**
 * Implements all the logic to perform K-means on a set of elements
 */
template <int k, int d>
class KMeansMPI
{
public:
    // some type definitions to make things easier
    typedef array<u_char, d> Element;
    class Cluster;
    typedef array<Cluster, k> Clusters; // define a cluster class to include all the things we need for a cluster. Has k clusters
    const int MAX_FIT_STEPS = 300;
    int rank = 0;

    // debugging for MPI, print out stuff
    const bool VERBOSE = false; // set to true for debugging output
#define V(stuff)             \
    if (VERBOSE)             \
    {                        \
        using namespace std; \
        stuff                \
    }

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()

     */
    virtual const Clusters &getClusters()
    {
        return clusters;
    }

    /**
     * fit() is the main k-means algorithm
     */
    virtual void fit(const Element *data, int data_n)
    {
        // init step, store all input data in elements
        elements = data;
        n = data_n; // calculate n datapoint distances
        fitWork(RootProcess);
    }

    /**
     * fitWork() is the function for each process to call to apply k-means on each process's partition
     * @param _rank - process rank
     */
    virtual void fitWork(int _rank)
    {
        rank = _rank;
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);               // collect number of processes so we can split into chunks to handle distances
        MPI_Bcast(&n, 1, MPI_INT, RootProcess, MPI_COMM_WORLD); // broadcast data & seed clusters to other processes
        processLengthPerProcess();                              // calculate length per process
        partition = (Element *)malloc(length_per_processes * d * sizeof(u_char));
        scatterElements(); // scatter elements to individual processes
        if (rank == RootProcess)
        {
            reseedClustersFixed(); // only the root can initialize the cluster seeds because hes has the full element list
        }
        bcastCentroids();                  // a separate function to broadcast centroids after re-seeding
        dist.resize(length_per_processes); // since when initializing, we dont know the size of the list of colors. This function is used to resize the 2D array based on n
        Clusters prior = clusters;
        prior[0].centroid[0]++; // just to make it different the first time
        int generation = 0;
        // step 4
        while (generation++ < MAX_FIT_STEPS && prior != clusters)
        {
            updateDistances(); // step 2
            prior = clusters;
            updateClusters();
            mergeClusters();  // merge cluster element indexes & calculate the new centroids
            bcastCentroids(); // broadcast the new centroids to all processes
        }
        free(partition);
    }

    /**
     * split elements from ROOT to partitions based on the process rank
     */
    virtual void scatterElements()
    {
        u_char *sendbuf = nullptr, *recvbuf = nullptr; // nullptr allows delete to work for anyone
        int *sendcounts = nullptr, *displs = nullptr;
        int m = 0; // size of partition

        if (rank == RootProcess)
        {
            // marshal data into sendbuf and set up sending side of message (RootProcess only)
            sendbuf = new u_char[n * d]; // max size
            sendcounts = new int[numProcs];
            displs = new int[numProcs];
            int i = 0; // index into sendbuf
            for (int pi = 0; pi < numProcs; pi++)
            {
                displs[pi] = i;
                // setup start & end index for the partition
                int begin_bucket = length_per_processes * pi;
                int end_bucket = begin_bucket + length_per_processes;
                // if it's the final process then the end index is n, no need to calculate further
                if (pi == numProcs - 1)
                    end_bucket = n;
                // assign each partition the corresponding element chunks
                for (int bi = begin_bucket; bi < end_bucket; bi++)
                {
                    for (const auto dimension : elements[bi])
                    {
                        sendbuf[i++] = dimension;
                    }
                }
                sendcounts[pi] = i - displs[pi];
            }
        }

        // set this->m for my process
        m = length_per_processes;

        // set up receiving side of message (everyone)
        int recvcount = m * d;
        recvbuf = new u_char[recvcount];

        MPI_Scatterv(sendbuf, sendcounts, displs, MPI_UNSIGNED_CHAR,
                     recvbuf, recvcount, MPI_UNSIGNED_CHAR,
                     RootProcess, MPI_COMM_WORLD);

        // unmarshal data from recvbuf into this->partion
        int j = 0; // index into recvbuf
        for (int bi = 0; bi < m; bi++)
        {
            Element element = Element{};
            // array<u_char, d> element;
            for (int di = 0; di < d; di++)
            {
                element[di] = recvbuf[j++];
            }
            partition[bi] = element;
        }
        // elements = partition;

        // free temp arrays
        if (rank == RootProcess)
        {
            delete[] sendbuf;
            delete[] sendcounts;
            delete[] displs;
        }
        delete[] recvbuf;
    }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster
    {
        Element centroid; // the current center (mean) of the elements in the cluster
        vector<int> elements;

        // equality is just the centroids, regarless of elements
        friend bool operator==(const Cluster &left, const Cluster &right)
        {
            return left.centroid == right.centroid; // equality means the same centroid, regardless of elements
        }
    };

protected:
    const Element *elements = nullptr; // set of elements to classify into k categories (supplied to latest call to fit())
    Element *partition = nullptr;      // set of elements to classify into k categories (supplied to latest call to fit())
    int n = 0;                         // number of elements in this->elements
    Clusters clusters;                 // k clusters resulting from latest call to fit()
    vector<array<double, k>> dist;     // dist[i][j] is the distance from elements[i] to clusters[j].centroid
    vector<int> seeds;                 // seed used to reseed the centroids
    int numProcs = 0;                  // number of processes
    int length_per_processes = 0;      // total partition size for each process

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters()
    {
        vector<int> candidates(n);
        iota(candidates.begin(), candidates.end(), 0);
        auto random = mt19937{random_device{}()};
        // Note that we need C++20 for sample
        sample(candidates.begin(), candidates.end(), back_inserter(seeds), k, random);
        for (int i = 0; i < k; i++)
        {
            clusters[i].centroid = elements[seeds[i]]; // randomly assign an element at index random to be centroid
            clusters[i].elements.clear();              // reset all elements to get new ones for this round
        }
    }

    // /**
    //  * Get the initial cluster centroids.
    //  * Default implementation here is to just pick k elements at random from the element
    //  * set
    //  * @return list of clusters made by using k random elements as the initial centroids
    //  */
    virtual void reseedClustersFixed()
    {
        for (int i = 0; i < k; i++)
        {
            seeds.push_back(i);
            clusters[i].centroid = elements[seeds[i]]; // randomly assign an element at index random to be centroid
            clusters[i].elements.clear();              // reset all elements to get new ones for this round
        }
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     */
    virtual void updateDistances()
    {
        for (int i = 0; i < length_per_processes; i++)
        {
            V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x ", partition[i][x]);)
            for (int j = 0; j < k; j++)
            {
                dist[i][j] = distance(clusters[j].centroid, partition[i]);
                V(cout << " " << dist[i][j];)
            }
            V(cout << endl;)
        }
    }

    /**
     * Recalculate the current clusters based on the new distances shown in this->dist.
     * Find out what elements belong to the cluster, and calculate the average distance to get the centroid
     */
    virtual void updateClusters()
    {
        // reinitialize all the clusters
        for (int j = 0; j < k; j++)
        {
            clusters[j].centroid = Element{};
            clusters[j].elements.clear();
        }
        // for each element, put it in its closest cluster (updating the cluster's centroid as we go)
        for (int i = 0; i < length_per_processes; i++)
        {
            int min = 0;
            for (int j = 1; j < k; j++)
                if (dist[i][j] < dist[i][min])
                    min = j;
            // find out the absolute index of the element that is in the ROOT process. i is the relative position
            // rank * length_per_processes will help jump to the correct partition position in the element list of ROOT.
            int trueElementIndex = i + rank * length_per_processes;
            if (rank == numProcs - 1)
            {
                trueElementIndex = i + rank * (n / numProcs); // reset back to only n / numProcs because the last process's length may be larger, using it can lead to index out of bound.
            }
            clusters[min].elements.push_back(trueElementIndex);
        }
    }

    /**
     * Broadcast the calculated centroids from ROOT to other processes for the next round
     */
    virtual void bcastCentroids()
    {
        int count = k * d; // each centroid is an element type of d dimensions. We have k clusters so the size is k * d
        u_char *buf = new u_char[count];
        if (rank == RootProcess)
        {
            int i = 0;
            for (int ki = 0; ki < k; ki++)
            {
                for (int di = 0; di < d; di++)
                {
                    buf[i++] = clusters[ki].centroid[di];
                }
            }
        }
        MPI_Bcast(buf, count, MPI_UNSIGNED_CHAR, RootProcess, MPI_COMM_WORLD);
        if (rank != RootProcess)
        {
            int i = 0;
            for (int ki = 0; ki < k; ki++)
            {
                Element centroid = Element{};
                for (int di = 0; di < d; di++)
                {
                    centroid[di] = buf[i++];
                }
                clusters[ki].centroid = centroid; // each process updates their centroid
            }
        }
        delete[] buf;
    }

    /**
     * Merge the element indexes and calculate the centroids based on the elements in each cluster.
     */
    virtual void mergeClusters()
    {
        int *sendbuf = nullptr, *recvbuf = nullptr; // nullptr allows delete to work for anyone
        int *recvcounts = nullptr, *displs = nullptr;
        int length = k * n;        // assume that each cluster has the maximum of n elements. hence we have length = k * n;
        sendbuf = new int[length]; // we gather all the cluster element sizes & send to the root first
        int i = 0;
        if (rank == RootProcess)
        {
            // has numProcs processes sending to the root
            recvbuf = new int[numProcs * length];
            recvcounts = new int[numProcs];
            displs = new int[numProcs];
            for (int pi = 0; pi < numProcs; pi++)
            {
                displs[pi] = pi * length;
                recvcounts[pi] = length;
            }
        }
        int totalElementSize = 0;
        for (int ki = 0; ki < k; ki++)
        {
            int size = clusters[ki].elements.size();
            totalElementSize += size;
            for (int ei = 0; ei < size; ei++)
            {
                sendbuf[i++] = clusters[ki].elements[ei];
            }
            // since we dont know the exact size of each cluster, we fill -1 in to make it full
            for (int ei = size; ei < n; ei++)
            {
                sendbuf[i++] = -1; // extra data to let all clusters have the same element indexes.
            }
        }
        MPI_Gatherv(sendbuf, length, MPI_UNSIGNED, recvbuf, recvcounts, displs, MPI_UNSIGNED, RootProcess, MPI_COMM_WORLD);
        if (rank == RootProcess)
        {

            // clear all elements to reset everything
            for (int j = 0; j < k; j++)
            {
                clusters[j].elements.clear();
            }
            int i = 0;
            for (int pi = 0; pi < numProcs; pi++)
            {
                for (int ki = 0; ki < k; ki++)
                {
                    for (int ni = 0; ni < n; ni++)
                    {
                        // if it is not a dummy value then it is the true element index, then use it
                        if (recvbuf[i] != -1)
                        {
                            // accumulate the centroid of that cluster.
                            // The order of the recvbuf: p0-k0-n0 p0-k0-n1...p0-k1-n0 ...p0-k9-n0...p32-k9-n10 if p = 32, k = 9 and n = 10
                            accum(clusters[ki].centroid, clusters[ki].elements.size(), elements[recvbuf[i]], 1);
                            clusters[ki].elements.push_back(recvbuf[i]);
                        }
                        i++; // always increment i to skip pass dummy values
                    }
                }
            }
        }

        // free temp arrays
        delete[] sendbuf;
        if (rank == RootProcess)
        {
            delete[] recvcounts;
            delete[] displs;
            delete[] recvbuf;
        }
    }

    /**
     * Method to update a centroid with an additional element(s)
     * @param centroid   accumulating mean of the elements in a cluster so far
     * @param centroid_n number of elements in the cluster so far
     * @param addend     another element(s) to be added; if multiple, addend is their mean
     * @param addend_n   number of addends represented in the addend argument
     */
    virtual void accum(Element &centroid, int centroid_n, const Element &addend, int addend_n) const
    {
        int new_n = centroid_n + addend_n;
        for (int i = 0; i < d; i++)
        {
            double new_total = (double)centroid[i] * centroid_n + (double)addend[i] * addend_n;
            centroid[i] = (u_char)(new_total / new_n);
        }
    }

    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element &a, const Element &b) const = 0;

    /**
     * Calculate element length of a process
     */
    void processLengthPerProcess()
    {
        length_per_processes = n / numProcs;

        // final process can handle more
        // add remaining if n / numProcs has remainings
        if (rank == numProcs - 1)
        {
            length_per_processes += n - length_per_processes * numProcs;
        }
    }
};
