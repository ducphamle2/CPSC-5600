/**
 * Based on Principles of Parallel Programming, Figure 7.6
 * (with corrections)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/*
 * Convergence parameters
 */
#define THRESHOLD   	1e-3f
#define MAX_ITERATIONS	1000

/*
 * Shape parameters
 * For the per-process matrix region, we have the cells
 * that we calculate: rows 1..Height-2 and columns 1..Width-2
 * and additional perimeter cells from neighboring processes
 * that we replicate locally: rows 0, Height-1, cols 0, Width-1
 * For regions on the edge, we fill in the perimeter cells
 * ourselves (1.0 for left edge, 0.0 for all others)
 */
#define DataRows 80
#define DataCols 20
#define Rows     4  // number of processes per column
#define Cols     2  // number of processes per row
#define Height   (DataRows/Rows + 2) // number of data rows per process
#define Width    (DataCols/Cols + 2) // number of data columns per process

/*
 * Edges
 */
#define Top    0
#define Left   0
#define Right  (Cols-1)
#define Bottom (Rows-1)

/*
 * Process Rank IDs
 */
#define RootProcess 0
#define NorthPE(i)  ((i) - Cols)
#define SouthPE(i)  ((i) + Cols)
#define EastPE(i)   ((i) + 1)
#define WestPE(i)   ((i) - 1)

/*
 * Inline Functions
 */
#define Swap(a,b)	{float (*tmp)[Width] = (a); (a) = (b); (b) = tmp;}
#define Abs(x)		((x) < 0.0 ? -(x) : (x))
#define Max(a,b)	((a) > (b) ? (a) : (b))

/*
 * Forward Declarations
 */
void gatherPrint(int, float, int, float (*)[Width]);
void printMyData(int, float (*)[Width], char *);

/**
 * Main entry point
 */
int main() {
	int numProcs, myID, row, col, i, j, k = 0, tag = 1;
	MPI_Status status;
	float delta, average, globalDelta = THRESHOLD;
	float one[Height][Width], two[Height][Width];
	float (*val)[Width], (*new)[Width];
	float buffer[Height-2];

	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	if (numProcs != Rows * Cols) {
		if (myID == RootProcess)
			printf("Must have exactly %d procs", Rows * Cols);
		exit(1);
	}
	row = myID / Cols;
	col = myID % Cols;
	
	// initialize val for this process
	val = one;
	new = two;
	memset(val, 0, Height * Width * sizeof(float));
	if (col == Left)
		for (i = 0; i < Height; i++)
			val[i][0] = 1.0f;
	memcpy(new, val, Height * Width * sizeof(float));
	// DBUG: for debugging, mark corners 
	// (which are not used in any calculation, so it's ok)
	// val[0][0]        = 1.11; val[0][Width-1]        = 2.22; 
	// val[Height-1][0] = 3.33; val[Height-1][Width-1] = 4.44;
	// new[0][0]        = 5.55; new[0][Width-1]        = 6.66; 
	// new[Height-1][0] = 7.77; new[Height-1][Width-1] = 8.88;
	
	do {
		/*
		 * Send perimeter of calculated cells to four neighbors
		 * Rows 1 and Height-2, Columns 1 and Width-2
		 */
		if (row != Top)
			MPI_Send(&val[1][1], Width-2, MPI_FLOAT, 
					 NorthPE(myID), tag, MPI_COMM_WORLD);
		if (col != Right) {
			for (i = 1; i < Height-1; i++)
				buffer[i-1] = val[i][Width-2];
			MPI_Send(buffer, Height-2, MPI_FLOAT, 
					 EastPE(myID), tag, MPI_COMM_WORLD);
		}
		if (row != Bottom)
			MPI_Send(&val[Height-2][1], Width-2, MPI_FLOAT, 
					 SouthPE(myID), tag, MPI_COMM_WORLD);
		if (col != Left) {
			for (i = 1; i < Height-1; i++)
				buffer[i-1] = val[i][1];
			MPI_Send(buffer, Height-2, MPI_FLOAT, 
					 WestPE(myID), tag, MPI_COMM_WORLD);
		}

		/*
		 * Receive perimeter of input cells from four neighbors
		 * Rows 0 and Height-1, Columns 0 and Width-1
		 */
		if (row != Top)
			MPI_Recv(&val[0][1], Width-2, MPI_FLOAT,
					 NorthPE(myID), tag, MPI_COMM_WORLD, &status);
		if (col != Right) {
			MPI_Recv(&buffer, Height-2, MPI_FLOAT,
					 EastPE(myID), tag, MPI_COMM_WORLD, &status);
			for (i = 1; i < Height-1; i++)
				val[i][Width-1] = buffer[i-1];
		}
		if (row != Bottom)
			MPI_Recv(&val[Height-1][1], Width-2, MPI_FLOAT,
					 SouthPE(myID), tag, MPI_COMM_WORLD, &status);
		if (col != Left) {
			MPI_Recv(&buffer, Height-2, MPI_FLOAT,
					 WestPE(myID), tag, MPI_COMM_WORLD, &status);
			for (i = 1; i < Height-1; i++)
				val[i][0] = buffer[i-1];
		}

		/*
		 * Calculate average, delta for all points
		 */
		delta = 0.0f;
		for (i = 1; i < Height-1; i++)
			for (j = 1; j < Width-1; j++) {
				float diff;
				average = (val[i-1][j] + val[i][j+1] 
							+ val[i+1][j] + val[i][j-1]) / 4;
				delta = Max(delta, Abs(average - val[i][j]));
				/* DBUG:
				diff = Abs(average - val[i][j]);
				if (diff > delta) {
					printf("g%03d [p%d][%d][%d] from %.3f to %.3f (d: %.6f)\n", 
							k, myID, i, j, val[i][j], average, diff);
					delta = diff;
				} DBUG */
				new[i][j] = average;
			}
		// DBUG: printf("g%03d %.6f\n", myID, delta);

		/*
		 * Find maximum diff (and send to everyone)
		 */
		MPI_Allreduce(&delta, &globalDelta, 1, MPI_FLOAT, MPI_MAX,
					  MPI_COMM_WORLD);
		// DBUG: if (myID == RootProcess) printf("R %.6f\n", globalDelta);
	
		/*
		 * Prepare for next generation
		 */
		Swap(val, new);
		k++;
		// DBUG: if(myID == RootProcess) 
		// 			printMyData(myID, val, "val (after swap)");
		// DBUG: gatherPrint(myID, globalDelta, k, val);

	} while (globalDelta >= THRESHOLD && k < MAX_ITERATIONS);

	// Print out result
	gatherPrint(myID, globalDelta, k, val);
	
	// Would need the following if we didn't use MPI_Allreduce 
	// above to send the globalDelta around to everyone
	//MPI_Abort(MPI_COMM_WORLD, 0);  // stop the others
	
	MPI_Finalize();
	return 0;
}

/**
 * Print out my region of the SOR matrix
 * (including overlap edges).
 * @param myID  my rank in MPI_COMM_WORLD
 * @param val   this proc's region of the SOR matrix
 * @param s     label for printout
 */
void printMyData(int myID, float (*val)[Width], char *s) {
	int r, c;
	printf("%d %s:\n", myID, s);
	for (r = 0; r < Height; r++) {
		if (r == 0 || r == Height-1)
			printf("* ");
		else
			printf("  ");
		for (c = 0; c < Width; c++) {
			if (c == 1 || c == Width-1)
				printf("| ");
			printf("%.3f ", val[r][c]);
		}
		printf("\n");
	}
}

/**
 * Print the entire SOR matrix by gathering all the data
 * from the individual processes and printing it out row by row.
 * @param myID        rank in MPI_COMM_WORLD
 * @param globalDelta last biggest element change for any elem
 * @param numIter     number of iterations done so far
 * @param myData      this proc's region of the SOR matrix
 */
void gatherPrint(int myID, float globalDelta, 
				 int numIter, float (*myData)[Width]) {
	int tag = 2;
	MPI_Status status;
	float *bigData = NULL;

	if (myID == RootProcess)
		bigData = (float*) malloc(Height * Width * Rows * Cols * sizeof(float));

	MPI_Gather(myData, Height * Width, MPI_FLOAT,
			   bigData, Height * Width, MPI_FLOAT,
			   RootProcess, MPI_COMM_WORLD);

	if (myID == RootProcess) {
		int r, c, dr, dc, hw = Height*Width;

		printf("max diff: %g, iterations: %d\n", globalDelta, numIter);
		
		// Get each proc's result and print it out in the right order
		// We get all the results from the first "Row" of processes, then print
		// those out, then get all the results from the next row of processes,
		// etc.
		for (c = (Width-2)*Cols; c > 0; c--) {
			printf("------");
			if (c % (Width-2) == 1)
				printf("+-");
		}
		printf("\n");
		for (r = 0; r < Rows; r++) {
			for (dr = 1; dr < Height-1; dr++) {
				for (c = 0; c < Cols; c++) {
					for (dc = 1; dc < Width-1; dc++)
						printf("%.3f ", 
							   bigData[hw*(Cols*r + c) + Width*dr + dc]);
					printf("| ");
				}
				printf("\n");
			}
			for (c = (Width-2)*Cols; c > 0; c--) {
				printf("------");
				if (c % (Width-2) == 1)
					printf("+-");
			}
			printf("\n");
		}
	}
}
