#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define RootProcess 0

/*
 * In this version, instead of multiple Sends use Scatter
 */

int main() {
	int *myArray, *bigArray = NULL;
	int length = 100;
	int length_per_process;

	int myID, value, numProcs;
	MPI_Status status;
	int myCount = 0, globalCount = 0;
	int i, p;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);

	length_per_process = (length-1) / numProcs + 1;
	myArray = (int*) malloc(length_per_process * sizeof(int));

	// Read the data, distribute it among the various processes
	if (myID == RootProcess) {
		FILE *fp;
		printf("l_p_p: %d, padded: %d\n", length_per_process, numProcs*length_per_process);
		if ((fp = fopen("popp3.dat", "r")) == NULL) {
			printf("fopen failed on popp3.dat");
			exit(1);
		}
		bigArray = (int*) malloc(numProcs * length_per_process * sizeof(int));
		for (i = 0; i < length; i++)
			fscanf(fp, "%d", bigArray + i);
		fclose(fp);
		for (i = length; i < numProcs * length_per_process; i++)
			bigArray[i] = 0;  // fill in any padding with non-3 value
	}
	MPI_Scatter(bigArray, length_per_process, MPI_INT, 
				myArray, length_per_process, MPI_INT,
				RootProcess, MPI_COMM_WORLD);
	
	// Do my actual work
	for (i = 0; i < length_per_process; i++)
		if (myArray[i] == 3)
			myCount++;  // Update local count
	
	MPI_Reduce(&myCount, &globalCount, 1, MPI_INT, MPI_SUM,
			   RootProcess, MPI_COMM_WORLD);

	if (myID == RootProcess)
		printf("Number of 3's: %d\n", globalCount);
	MPI_Finalize();
	return 0;
}
