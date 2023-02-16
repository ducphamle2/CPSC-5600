Color.h

3 colors RGB, 8 bits - consider as an array of 3 numbers / array of 8-bit numbers
<u_char, 3> - 3 bytes in the array

Colors are basically (n=4):

|R value in bytes|G value in bytes|B value in bytes| index 0|=>P1 chunk
|R value in bytes|G value in bytes|B value in bytes| index 1|=>P1 chunk
|R value in bytes|G value in bytes|B value in bytes| index 2|=>P2 chunk
|R value in bytes|G value in bytes|B value in bytes| index 3|=>P2 chunk

Calculate the distance of each element to the centroid of each cluster. Set the element to the closest distance cluster centroid.

How to do parallel in hw5: split the list of colors into different process chunks. Each chunk handles a set of elements in the list. Calculate the distances, then we calculate the centroids. Also, needs to parallel the updateClusters(), where each cluster gets the elements & calculate their averages to get the new centroid (this new centroid is used in the next loop). These steps can be parallel.
We can reduce after calculating element distances (first parallel), and after calculating the new avarage to get the new centroid for a cluster (2nd parallel). We need to wait for the new centroid (the first list of centroids are chosen randomly) to calculate the distances again, and after getting the distance, we again calculate the centroids.

Start with 2 processes, small datasets & print out the data

Extra credit: code neu da chay thi se chay cho extra credit

MPI Review:

general
    MPI_Init
    MPI_Finalize
    MPI_Comm_Size
    MPI_Comm_Rank
    MPI_Wtime

communication
    point-to-point
        MPI_Send
        MPI_Recv
    collective-routine
        MPI_Barrier
        MPI_BCast
        MPI_Scatter / MPI_Scatterv - v for vector
        MPI_Gather / MPI_Gatherv
        MPI_Reduce
        MPI_Scan

Signal: Aborted / Signal code ; Corrupted size vs prev size => most likely memory error. How to debug: use valgrind. How to: mpirun -n 6 valgrind ./popp71