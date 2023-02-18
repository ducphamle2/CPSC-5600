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
    virtual void fitWork(int rank)
    {
        // broadcast data & seed clusters to other processes
        MPI_Bcast(&n, 1, MPI_INT, RootProcess, MPI_COMM_WORLD);
        bcastElements(rank);
        // MPI_Bcast(&elements, n, MPI_UNSIGNED_CHAR, RootProcess, MPI_COMM_WORLD); // has n number of color elements, so size is n
        cout << "n: " << n << endl;
        cout << "rank: " << rank << " element size: " << elements->size() << endl;
        if (rank == RootProcess)
        {
            reseedClusters(); // find random values to get started, step 1
        }
        bcastSeeds(rank); // need a separate function to broadcast centroids after re-seeding
        dist.resize(n);   // since when initializing, we dont know the size of the list of colors. This function is used to resize the 2D array based on n

        cout << "rank: " << rank << " before printing cluster address" << endl;
        cout << "rank: " << rank << " clusters address: " << &clusters << endl;

        Clusters prior = clusters;
        prior[0].centroid[0]++; // just to make it different the first time
        int generation = 0;
        // step 4
        while (generation++ < MAX_FIT_STEPS && prior != clusters)
        {
            updateDistances(); // step 2
            prior = clusters;
            updateClusters();
        }
    }

    virtual void bcastElements(int rank)
    {
        char *buf = (char *)malloc(n * d * sizeof(char));
        if (rank == RootProcess)
        {
            int i = 0;
            for (int j = 0; j < n; j++)
            {
                for (int jd = 0; jd < d; jd++)
                {
                    buf[i++] = elements[j][jd];
                }
            }
        }
        MPI_Bcast(buf, n * d, MPI_UNSIGNED_CHAR, RootProcess, MPI_COMM_WORLD);
        if (rank != RootProcess)
        {
            int i = 0;
            Element *data = (Element *)malloc(n * d * sizeof(char));
            for (int j = 0; j < n; j++)
            {
                Element element = Element{};
                for (int jd = 0; jd < d; jd++)
                {
                    element[jd] = buf[i++];
                }
                data[j] = element;
            }
            elements = data;
        }
        delete[] buf;
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

    // /**
    //  * Calculate the distance from each element to each centroid.
    //  * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
    //  */
    // virtual void updateDistancesMPI(int rank)
    // {

    //     MPI_Comm_size(MPI_COMM_WORLD, &numProcs); // collect number of processes so we can split into chunks to handle distances
    //     double *sendbuf;
    //     double *recvbuf;
    //     std::vector<std::array<double, k>> *partition;
    //     int *sendcounts = NULL; // for scatterv
    //     int *displs = NULL;     // for scatterv
    //     int m;                  // size of partition
    //     int length_per_processes = n / numProcs;

    //     // final process can handle more
    //     // add remaining if n / numProcs has remainings
    //     if (rank == numProcs - 1)
    //     {
    //         length_per_processes += n - length_per_processes * numProcs;
    //     }

    //     // only root process will scatter
    //     if (rank == RootProcess)
    //     {
    //         sendcounts = (int *)malloc(length_per_processes * sizeof(int));
    //         displs = (int *)malloc(length_per_processes * sizeof(int));
    //         sendbuf = (double *)malloc(n * d * k * sizeof(double)); // each array has d elements, for color is RGB; we need to calculate the array distance against k clusters; we have n elements. We dont need the size of each dist because it is always k

    //         for (int i = 0; i < numProcs; i++)
    //         {
    //             sendcounts[i] = length_per_processes;
    //             displs[i] = i * length_per_processes;
    //         }
    //         // last send count may be too small, so fix it
    //         sendcounts[numProcs - 1] = n - (numProcs - 1) * length_per_processes;
    //     }

    //     MPI_Scatterv(sendbuf, sendcounts, displs, MPI_DOUBLE, recvbuf, length_per_processes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //     // TODO: need to free resources

    //     // TODO: need to scatterv here so that each process receive a chunk of the bucket
    //     // each bucket has a list of arrays (each array has length k - because we always calculate distances for k clusters) with bucket length is "length_per_process". Length of root is "root_length"
    //     // Afterwards, we need to Allgatherv to collect the dist for each process. Each process needs complete identical dist so that it can update its clusters later on

    //     // if (rank == RootProcess) // root case, we resize the dist + remaining length if divisor has remainings
    //     // {
    //     //     for (int i = 0; i < root_length; i++)
    //     //     {
    //     //         V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x", elements[i][x]); cout << endl;)
    //     //         for (int j = 0; j < k; j++)
    //     //         {
    //     //             dist[i][j] = distance(clusters[j].centroid, elements[i]);
    //     //             V(cout << " " << dist[i][j];)
    //     //         }
    //     //         V(cout << endl;)
    //     //     }
    //     // }
    //     // else
    //     // {
    //     //     for (int i = 0; i < length_per_processes; i++)
    //     //     {
    //     //         V(cout << "distances for " << i << "("; for (int x = 0; x < d; x++) printf("%02x", elements[i][x]); cout << endl;)
    //     //         for (int j = 0; j < k; j++)
    //     //         {
    //     //             dist[i][j] = distance(clusters[j].centroid, elements[i]);
    //     //             V(cout << " " << dist[i][j];)
    //     //         }
    //     //         V(cout << endl;)
    //     //     }
    //     // }
    //     // // All gather to receive complete dist
    //     // int gather_length = rank == 0 ? root_length : length_per_processes;
    //     // MPI_Allgatherv(processDist, gather_length, MPI_UNSIGNED_CHAR, &dist, gather_length, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
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
};
