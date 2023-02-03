/*
 * Le Duc Pham
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

/**
 * @class BitonicSequential class per cpsc5600 hw3 specification.
 * @versioon 24-Jan-2020
 */
public class BitonicSequential {
    public static final int N = 1 << 22; // size of the final sorted array (power of two)
    public static final int TIME_ALLOWED = 10; // seconds

    /**
     * swap - swap two array elements of the instance variable bitonicSeq
     * 
     * @param i - first index to swap
     * @param j - second index to swap
     */
    private static void swap(double[] bitonicSeq, int i, int j) {
        double temp = bitonicSeq[i];
        bitonicSeq[i] = bitonicSeq[j];
        bitonicSeq[j] = temp;
    }

    /**
     * Helper for printing out bits. Converts the last four bits of the given number
     * to a string of 0's and 1's.
     * 
     * @param n number to convert to a string (only last four bits are observed)
     * @return four-character string of 0's and 1's
     */
    private static String fourbits(int n) {
        String ret = /* to_string(n) + */(n > 15 ? "/1" : "/");
        for (int bit = 3; bit >= 0; bit--) {
            int check = n & 1 << bit;
            ret += check;
        }
        return ret;
    }

    /**
     * Recursively merge and sort the bitonic sequence. The result of this function
     * is a sorted array based on the given direction
     * 
     * @param n         size of the array
     * @param direction direction to sort, can be either up or down
     */
    private static void bitonicSort(double[] seq, int n) {
        // k is size of the pieces, starting at pairs and doubling up until we get to
        // the whole array
        // k also determines if we want ascending or descending for each section of i's
        // corresponds to 1<<d in textbook
        for (int k = 2; k <= n; k *= 2) { // k is one bit, marching to the left
            // System.out.printf("%s\t", fourbits(k));

            // j is the distance between the first and second halves of the merge
            // corresponds to 1<<p in textbook
            for (int j = k / 2; j > 0; j /= 2) { // j is one bit, marching from k to the right
                // if (j != k / 2)
                // System.out.printf(" \t");
                // System.out.printf("%s\t", fourbits(j));

                // i is the merge element
                for (int i = 0; i < n; i++) {
                    // if (i != 0)
                    // System.out.printf(" \t \\t");
                    // System.out.printf("%s\t", fourbits(i));

                    int ixj = i ^ j; // xor: all the bits that are on in one and off in the other
                    // System.out.printf("%s\t%s\n", fourbits(ixj), fourbits(i & k));

                    // only compare if ixj is to the right of i
                    if (ixj > i) {
                        if ((i & k) == 0 && seq[i] > seq[ixj])
                            swap(seq, i, ixj);
                        if ((i & k) != 0 && seq[i] < seq[ixj])
                            swap(seq, i, ixj);
                    }
                }
            }
        }
    }

    /**
     * Main entry for HW3 assignment.
     *
     * @param args not used
     */
    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        int work = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double[] data = RandomArrayGenerator.getArray(N);
            // Note that BitonicStage assumes both its input arrays are sorted
            // increasing. It then inverts its second input to form a true bitonic
            // sequence from the concatenation of the first input with the inverted
            // second input.
            bitonicSort(data, N);
            if (!RandomArrayGenerator.isSorted(data) || N != data.length) {
                System.out.println("failed");
                System.exit(1);
            }
            work++;
        }
        System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in "
                + TIME_ALLOWED + " seconds");
    }
}
