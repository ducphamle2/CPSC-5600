/**
 * @file KMeans.h - implementation of k-means clustering algorithm
 * @author Kevin Lundeen
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#pragma once // only process the first time it is included; ignore otherwise
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <set>
#include <array>
#include "mpi.h"
using namespace std;

#define RootProcess 0

// d here is the dimensions of our data. For RGB colors it is 3
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
    const bool VERBOSE = true; // set to true for debugging output
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
     * fit() is the main k-means algorithm
     */
    virtual void fitWork(int _rank)
    {
        rank = _rank;
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);               // collect number of processes so we can split into chunks to handle distances
        processLengthPerProcess(rank);                          // calculate length per process
        MPI_Bcast(&n, 1, MPI_INT, RootProcess, MPI_COMM_WORLD); // broadcast data & seed clusters to other processes
        scatterElements();
        if (rank == RootProcess)
        {
            reseedClusters(); // find random values to get started, step 1
        }
        bcastSeeds(rank); // need a separate function to broadcast centroids after re-seeding
        dist.resize(n);   // since when initializing, we dont know the size of the list of colors. This function is used to resize the 2D array based on n
        Clusters prior = clusters;
        prior[0].centroid[0]++; // just to make it different the first time
        int generation = 0;
        // step 4
        while (generation++ < MAX_FIT_STEPS && prior != clusters)
        {
            updateDistances(); // step 2
            prior = clusters;
            updateClusters();
            // mergeClusters();
        }
    }

    virtual void scatterElements()
    {
        u_char *sendbuf = nullptr, *recvbuf = nullptr; // nullptr allows delete to work for anyone
        int *sendcounts = nullptr, *displs = nullptr;
        int elements_per_proc = n / numProcs;
        int m = 0; // size of partition
        Element *partition = nullptr;

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
                int begin_bucket = elements_per_proc * pi;
                int end_bucket = begin_bucket + elements_per_proc;
                if (pi == numProcs - 1)
                    end_bucket += n - elements_per_proc * numProcs; // extras for last proc
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
        m = elements_per_proc;
        if (rank == numProcs - 1)
            m += n - elements_per_proc * numProcs;

        // set up receiving side of message (everyone)
        int recvcount = m * d;
        recvbuf = new u_char[recvcount];

        MPI_Scatterv(sendbuf, sendcounts, displs, MPI_UNSIGNED_CHAR,
                     recvbuf, recvcount, MPI_UNSIGNED_CHAR,
                     RootProcess, MPI_COMM_WORLD);

        // unmarshal data from recvbuf into this->partion
        partition = (Element *)malloc(m * d * sizeof(char)); // calls default ctor for each
        int j = 0;                                           // index into recvbuf
        for (int bi = 0; bi < m; bi++)
        {
            int esize = recvbuf[j++];
            for (int ei = 0; ei < esize; ei++)
            {
                Element element = Element{};
                for (int di = 0; di < d; di++)
                {
                    element[di] = recvbuf[j++];
                }
                partition[bi] = element;
            }
        }
        elements = partition;

        // free temp arrays
        delete[] sendbuf;
        delete[] sendcounts;
        delete[] partition;
        delete[] displs;
        delete[] recvbuf;
    }

    virtual void bcastSeeds(int rank)
    {
        int *buf = (int *)malloc(k * sizeof(int));
        if (rank == RootProcess)
        {
            for (int i = 0; i < k; i++)
            {
                buf[i] = seeds[i];
            }
        }
        MPI_Bcast(buf, k, MPI_INT, RootProcess, MPI_COMM_WORLD);
        if (rank != RootProcess)
        {
            for (int i = 0; i < k; i++)
            {
                clusters[i].centroid = elements[buf[i]];
                clusters[i].elements.clear();
            }
        }
        delete[] buf;
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
    int n = 0;                         // number of elements in this->elements
    Clusters clusters;                 // k clusters resulting from latest call to fit()
    vector<array<double, k>> dist;     // dist[i][j] is the distance from elements[i] to clusters[j].centroid
    vector<int> seeds;                 // seed used to reseed the centroids
    int numProcs = 0;                  // number of processes
    int length_per_processes = 0;

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
            cout << "seed[i] in root: " << seeds[i] << endl;
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
        for (int i = 0; i < n; i++)
        {
            V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x", elements[i][x]);)
            for (int j = 0; j < k; j++)
            {
                dist[i][j] = distance(clusters[j].centroid, elements[i]);
                V(cout << " " << dist[i][j];)
            }
            V(cout << endl;)
        }
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     */
    // virtual void updateDistancesProcesses()
    // {
    //     int slice = n / numProcs;
    //     int startIndex = rank * slice;
    //     // if id is not final thread, then move to the end of piece by adding 1, else
    //     // end is already at the last element of data
    //     int endIndex = rank != numProcs - 1 ? (rank + 1) * slice : n;
    //     for (int i = startIndex; i < endIndex; i++)
    //     {
    //         V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x", elements[i][x]);)
    //         for (int j = 0; j < k; j++)
    //         {
    //             dist[startIndex][j] = distance(clusters[j].centroid, elements[startIndex]);
    //             V(cout << " " << dist[startIndex][j];)
    //         }
    //         V(cout << endl;)
    //     }
    // }

    // /**
    //  * Calculate the distance from each element to each centroid.
    //  * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
    //  */
    // virtual void updateDistancesMPIGather(int rank)
    // {
    //     // vector<array<double, k>> distanceDist = new vector<array<double, k>>[length_per_processes];
    //     int sendbufIndex = 0;
    //     double *sendbuf = (double *)malloc(n * (1 + k) * sizeof(double)); // maximum length of sendbuf is n when numProcs is 1, so we use n instead of length_per_processes to reduce overhead
    //     double *recvbuf;
    //     int sendcounts = (int *)malloc(length_per_processes * sizeof(int));
    //     int displs = (int *)malloc(length_per_processes * sizeof(int));
    //     int i = 0; // size of sendbuf
    //     int startIndex = rank * length_per_processes;
    //     // if id is not final thread, then move to the end of piece by adding 1, else
    //     // end is already at the last element of data
    //     int endIndex = rank != numProcs - 1 ? (rank + 1) * length_per_processes : n;
    //     for (int i = startIndex; i < endIndex; i++)
    //     {
    //         V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x", elements[i][x]);)
    //         for (int j = 0; j < k; j++)
    //         {
    //             sendbuf[sendbufIndex++] = distance(clusters[j].centroid, elements[i]);
    //             V(cout << " " << sendbuf[sendbufIndex - 1];)
    //         }
    //         V(cout << endl;)
    //     }

    //     MPI_Allgatherv(sendbuf, sendbufIndex, MPI_DOUBLE, recvbuf,)
    // }

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
        for (int i = 0; i < n; i++)
        {
            int min = 0;
            // can parallel here because we are working on different clusters. They can work at the same time
            for (int j = 1; j < k; j++)
                if (dist[i][j] < dist[i][min])
                    min = j;
            accum(clusters[min].centroid, clusters[min].elements.size(), elements[i], 1);
            clusters[min].elements.push_back(i);
        }
    }

    // /**
    //  * Recalculate the current clusters based on the new distances shown in this->dist.
    //  * Find out what elements belong to the cluster, and calculate the average distance to get the centroid
    //  */
    // virtual void updateClustersMPI()
    // {
    //     // if id is not final thread, then move to the end of piece by adding 1, else
    //     // end is already at the last element of data
    //     int endIndex = rank != numProcs - 1 ? (rank + 1) * slice : n;
    //     // reinitialize all the clusters
    //     for (int j = 0; j < k; j++)
    //     {
    //         clusters[j].centroid = Element{};
    //         clusters[j].elements.clear();
    //     }
    //     // for each element, put it in its closest cluster (updating the cluster's centroid as we go)
    //     for (int i = startIndex; i < endIndex; i++)
    //     {
    //         int min = 0;
    //         // can parallel here because we are working on different clusters. They can work at the same time
    //         for (int j = 1; j < k; j++)
    //             if (dist[i][j] < dist[i][min])
    //                 min = j;
    //         clusters[min].elements.push_back(i);
    //     }
    // }

    // /**
    //  * Merge the clusters to the root process by using Gatherv
    //  */
    // virtual void mergeClusters()
    // {
    //     int slice = n / numProcs;
    //     int m = slice;
    //     if (rank == numProcs - 1)
    //         m += n - slice * numProcs;
    //     int startIndex = rank * slice;

    //     int *sendbufElements = nullptr, *recvbufElements = nullptr; // nullptr allows delete to work for anyone
    //     int *recvcounts = nullptr, *displs = nullptr;

    //     sendbufElements = new int[(n + 1) * k]; // +1 reserved for size of a cluster's elements
    //     int i = 0;

    //     for (int clusterIndex = 0; clusterIndex < k; clusterIndex++)
    //     {
    //         int elementSize = static_cast<int>(clusters[clusterIndex].size());
    //         sendbufElements[i++] = elementSize;
    //         for (int element : clusters[clusterIndex])
    //         {
    //             sendbufElements[i++] = element;
    //         }
    //     }
    //     recvcounts[rank] = i;
    //     if (rank == RootProcess)
    //     {
    //         // init recvBufElements
    //         recvbufElements = new int[(n + 1) * k]; // +1 reserved for size of a cluster's elements
    //         recvcounts = new int[numProcs];
    //         displs = new int[numProcs];
    //     }

    //     // gather elements of clusters
    //     MPI_Gatherv(sendbufElements, i, MPI_UNSIGNED, recvbufElements, recvcounts, displs, MPI_UNSIGNED, RootProcess, MPI_COMM_WORLD);
    //     MPI_AV

    //     // free temp arrays
    //     delete[] sendbufElements;
    //     delete[] recvcounts;
    //     delete[] displs;
    //     delete[] recvbufElements;
    // }

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
    void processLengthPerProcess(int rank)
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
