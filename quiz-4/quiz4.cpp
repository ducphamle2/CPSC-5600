#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <vector>
#define RootProcess 0

/*
 * In this version, instead of multiple Sends use Scatterv
 */

// note quiz 4

/**
 * Need to allocate how big sendBuf is? Can be exact size or max size
 * in this quiz, max size is: (1 + 2 * 5) * n  where n is number of buckets, 1 is reserved for displaying size of bucket, and 2 is total element of an array of the bucket, and 5 is maximum size of bucket
 * We have to flat out the bucket to make it an array & scatter it to other processes
 */

int scatterBuckets()
{
    int n;
    std::vector<std::array<double, k>> bucket;
    std::vector<std::array<double, k>> partition; // used for recvbuf
    int m;                                        // size for partition
    int p;
    const int MAX_BUCKET_SIZE;
    const int ROOT = 0;
    int length_per_process = n / p;
    int rank;
    int *sendbuf = NULL; // ROOT only allocation
    int *recvbuf;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == p - 1)
    {
        length_per_process += n - length_per_process * p;
    }
    recvbuf = (int *)malloc((1 + 2 * 5) * length_per_process * sizeof(int));

    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(p * sizeof(int)); // ROOT only allocation, other processes dont need this value
        displs = (int *)malloc(p * sizeof(int));     // ROOT only allocation, other processes dont need this value
        sendbuf = (int *)malloc((1 + 2 * 5) * n * sizeof(int));

        int sendBufIndex = 0;
        // fill sendbuf content
        for (int pi = 0; pi < p; pi++)
        {
            displs[pi] = sendBufIndex; // sendBufIndex is moving along
            int bi_begin = pi * length_per_process;
            int bi_end = bi_begin + length_per_process;
            if (pi == p - 1)
            {
                bi_end += n - p * length_per_process;
            }
            for (int bi = bi_begin; bi < bi_end; bi++)
            {
                sendbuf[sendBufIndex++] = bucket[bi].size();
                for (const ele : bucket[bi])
                { // this loop gets us an array in the bucket
                    for (const data : ele)
                    { // loop through the element of the array & add it into the sendbuf
                        sendbuf[sendBufIndex++] = data;
                    }
                }
            }
            sendcounts[pi] = sendBufIndex - displs[pi];
        }

        for (int i = 0; i < p; i++)
        {
            sendcounts[i] = length_per_process;
            displs[i] = i * length_per_process;
        }
    };
    m = (rank == p - 1) ? n - (p - 1) * length_per_process : length_per_process;
    partition = new vector<array<u_short, 2>>[m]; // the partition bucket we want the process P to convert to from the recvbuf
    int recvcount = m * (1 * 2 * MAX_BUCKET_SIZE);
    recvbuf = new int[recvcount]; // the array chunk received by process P
    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT, recvbuf, length_per_process, MPI_INT, 0, MPI_COMM_WORLD);

    // unmarshal after getting the scattered data
    // for example, P1 receives a chunk of sendbuf, need to reverse

    int j = 0;
    for (int bi = 0; bi < m; bi++)
    {
        int bucket_size = recvbuf[bi++];
        for (int ei = 0; ei < bucket_size; ei++)
        {
            u_short key = recvbuf[j++];
            u_short value = recvbuf[j++];
            partition[bi].push_back(new element);
        }
    }
}

int scatterOverlap()
{
    int n;
    const int OVERLAP = 2;
    vector *table;
    int p;
    const int ROOT = 0;
    int length_per_process = n / p + OVERLAP * 2;
    int rank;
    int *sendbuf = NULL; // ROOT only allocation
    int *recvbuf;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == p - 1)
    {
        length_per_process += n - length_per_process * p;
    }
    recvbuf = (int *)malloc((1 + 2 * 5) * length_per_process * sizeof(int));

    int *sendcounts = NULL;
    int *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(p * sizeof(int)); // ROOT only allocation, other processes dont need this value
        displs = (int *)malloc(p * sizeof(int));     // ROOT only allocation, other processes dont need this value
        sendbuf = (int *)malloc((1 + 2 * 5) * n * sizeof(int));
        for (int i = 0; i < p; i++)
        {
            sendcounts[i] = length_per_process; // multiply by 2 because get last two elements of previous
            if (i == 0)
                displs[i] = 0;
            else
                displs[i] = i * length_per_process;
        }
    };
    MPI_Scatterv(sendbuf, sendcounts, displs, MPI_UNSIGNED, recvbuf, length_per_process, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}
