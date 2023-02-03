import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.BrokenBarrierException;

/*
 * Le Duc Pham
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

/**
 * @class BitonicParallel class per cpsc5600 hw3 specification.
 * @versioon 24-Jan-2020
 */
public class BitonicParallel {
    public static final int N = 1 << 22; // size of the final sorted array (power of two)
    public static final int TIME_ALLOWED = 10; // seconds
    public static final int N_THREADS = 8;

    /**
     * Bitonic stage of a bitonic sorting network pipeline. This class takes in
     * two ASC sorted arrays as inputs
     * and sorts them using the bitonic sort algorithm.
     */
    static class BitonicLoopParallel implements Runnable {

        /**
         * Default constructor of Bitonic stage used by BitonicSequential
         */
        public BitonicLoopParallel(double[] seq, CyclicBarrier barrier, int start, int end) {
            this.seq = seq;
            this.barrier = barrier;
            this.start = start;
            this.end = end;
        }

        /**
         * swap - swap two array elements of the instance variable bitonicSeq
         * 
         * @param i - first index to swap
         * @param j - second index to swap
         */
        private void swap(int i, int j) {
            double temp = seq[i];
            seq[i] = seq[j];
            seq[j] = temp;
        }

        /**
         * The Runnable part of the class. Polls the input queue and when ready, process
         * (sort)
         * it and then write it to the output queue.
         */
        @Override
        public void run() {

            // k is size of the pieces, starting at pairs and doubling up until we get to
            // the whole array
            // k also determines if we want ascending or descending for each section of i's
            // corresponds to 1<<d in textbook
            for (int k = 2; k <= N; k *= 2) { // k is one bit, marching to the left
                // System.out.printf("%s\t", fourbits(k));

                // j is the distance between the first and second halves of the merge
                // corresponds to 1<<p in textbook
                for (int j = k / 2; j > 0; j /= 2) { // j is one bit, marching from k to the right
                    // if (j != k / 2)
                    // System.out.printf(" \t");
                    // System.out.printf("%s\t", fourbits(j));

                    // i is the merge element
                    for (int i = start; i < end; i++) {
                        // if (i != 0)
                        // System.out.printf(" \t \\t");
                        // System.out.printf("%s\t", fourbits(i));

                        int ixj = i ^ j; // xor: all the bits that are on in one and off in the other
                        // System.out.printf("%s\t%s\n", fourbits(ixj), fourbits(i & k));

                        // only compare if ixj is to the right of i
                        if (ixj > i) {
                            if ((i & k) == 0 && seq[i] > seq[ixj])
                                swap(i, ixj);
                            if ((i & k) != 0 && seq[i] < seq[ixj])
                                swap(i, ixj);
                        }
                    }

                    try {
                        barrier.await();

                    } catch (BrokenBarrierException | InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }

        /**
         * Helper for printing out bits. Converts the last four bits of the given number
         * to a string of 0's and 1's.
         * 
         * @param n number to convert to a string (only last four bits are observed)
         * @return four-character string of 0's and 1's
         */
        private String fourbits(int n) {
            String ret = /* to_string(n) + */(n > 15 ? "/1" : "/");
            for (int bit = 3; bit >= 0; bit--) {
                int check = n & 1 << bit;
                ret += check;
            }
            return ret;
        }

        private double[] seq;
        private CyclicBarrier barrier;
        private int start, end;

    }

    /**
     * Main entry for HW3 assignment.
     *
     * @param args not used
     */

    public static void dump(double[] arr) {
        for (double ele : arr) {
            System.out.printf("%2f ", ele);

        }
        System.out.println("");
    }

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        int work = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double[] data = RandomArrayGenerator.getArray(N);
            CyclicBarrier barrier = new CyclicBarrier(N_THREADS);
            Thread[] threads = new Thread[N_THREADS];
            for (int i = 0; i < threads.length; i++) {
                // BitonicParallel.dump(data);
                int piece = N / N_THREADS;
                // the first thread starts at 0 til the end of the piece, the 2nd thread starts
                // at the next pieice and so on.
                int startIndex = i * piece;
                // if id is not final thread, then move to the end of piece by adding 1, else
                // end is already at the last element of data
                int endIndex = i != N_THREADS - 1 ? (i + 1) * piece : N;
                threads[i] = new Thread(new BitonicLoopParallel(data, barrier, startIndex, endIndex));
                threads[i].start();
            }

            // join to begin validating the array. If we dont join here then the validation
            // will always fail because of the threads running in parallel
            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            // BitonicParallel.dump(ult);
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
