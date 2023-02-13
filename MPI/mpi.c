#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    int rank;
    int world;
    char *processor_name = (char *)malloc(sizeof(char) * MPI_MAX_PROCESSOR_NAME);
    int *length = (int *)malloc(sizeof(int) * MPI_MAX_PROCESSOR_NAME);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Get_processor_name(processor_name, length);
    printf("Hello from: %s, rank %d, world: %d\n", processor_name, rank, world);
    MPI_Finalize();
}