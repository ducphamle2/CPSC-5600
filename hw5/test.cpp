#include <iostream>
#include "mpi.h"
using namespace std;

int main()
{
    int rank;
    int size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int *buf = (int *)malloc(2 * sizeof(int));
    int *recvbuf;
    int *rcounts = nullptr;
    int *displs = nullptr;
    int sendcount = 0;
    if (rank == 0)
    {
        recvbuf = (int *)malloc(2 * size * sizeof(int));
        rcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    if (rank == size - 1)
    {
        buf[sendcount++] = 9;
    }
    else
    {
        buf[sendcount++] = 10;
        buf[sendcount++] = 11;
    }
    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            rcounts[i] = 2;
            displs[i] = i * 2;
            if (i == size - 1)
            {
                rcounts[i] = 1;
            }
        }
    }

    for (int i = 0; i < sendcount; i++)
    {
        cout << "rank: " << rank << " buf: " << buf[i] << endl;
    }

    MPI_Gatherv(buf, sendcount, MPI_UNSIGNED, recvbuf, rcounts, displs, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Test rank 0" << endl;
        for (int i = 0; i < 5; i++)
        {
            cout << recvbuf[i] << endl;
        }
        for (int i = 0; i < size; i++)
        {
            cout << "displ: " << displs[i] << endl;
        }
    }

    MPI_Finalize();
}