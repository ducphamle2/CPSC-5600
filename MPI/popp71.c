#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define RootProcess 0
int main()
{
	int *myArray;
	int length = 10;
	int length_per_process;

	int myID, value, numProcs;
	MPI_Status status;
	int myCount = 0, globalCount = 0;
	int i, p;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	printf("num procs: %d\n", numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);

	length_per_process = length / numProcs;
	myArray = (int *)malloc(length_per_process * sizeof(int));

	// Read the data, distribute it among the various processes
	if (myID == RootProcess)
	{
		FILE *fp;
		if ((fp = fopen("popp3.dat", "r")) == NULL)
		{
			printf("fopen failed on popp3.dat");
			exit(1);
		}
		for (p = 0; p < numProcs - 1; p++)
		{
			for (i = 0; i < length_per_process; i++)
				fscanf(fp, "%d", myArray + i); // the toInt
			printf("p: %d\n", p);
			MPI_Send(myArray, length_per_process, MPI_INT, p + 1,
					 1, MPI_COMM_WORLD);
		}
		// Now read my data (last segment of data)
		for (i = 0; i < length - (numProcs - 1) * length_per_process; i++)
			fscanf(fp, "%d", myArray + i);
	}
	else
	{
		MPI_Recv(myArray, length_per_process, MPI_INT, RootProcess,
				 1, MPI_COMM_WORLD, &status);
	}

	// Do the actual work
	for (i = 0; i < length_per_process; i++)
		if (myArray[i] == 3)
			myCount++; // Update local count

	MPI_Reduce(&myCount, &globalCount, 1, MPI_INT, MPI_SUM,
			   RootProcess, MPI_COMM_WORLD);

	if (myID == RootProcess)
		printf("Number of 3's: %d\n", globalCount);
	MPI_Finalize();
	return 0;
}
