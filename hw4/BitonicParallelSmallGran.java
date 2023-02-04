import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.BrokenBarrierException;

/*
 * Le Duc Pham
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

/**
 * @class BitonicParallelSmallGran class per cpsc5600 hw4 specification.
 * @versioon 24-Jan-2020
 */
public class BitonicParallelSmallGran {
    public static final int N = 1 << 22; // size of the final sorted array (power of two)
    public static final int TIME_ALLOWED = 10; // seconds
    public static final int N_THREADS = 8;
    public static final Granularity GRANULARITY = Granularity.MAX; // FIXME: change this to adjust the granularity
    // of
    // the algorithm

    public enum Granularity {
        MAX, // each j has one barrier. Base case
        HALF, // each j has one barrier in the middle, and one after finished
        QUARTER, // each j has 4 barriers
        NONE, // each comparator has a barrier
    }

    /**
     * A thread that does the bitonic loop from scratch, but only swaps a specific
     * number of elements. The idea is to have multiple threads running the exact
     * same code, but handling different parts of the array => data parallelism
     */
    static class BitonicLoopParallel implements Runnable {

        /**
         * Constructor of BitonicLoopParallel
         */
        public BitonicLoopParallel(double[] seq, CyclicBarrier barrier, Granularity granularity, int start, int end) {
            this.seq = seq;
            this.barrier = barrier;
            this.start = start;
            this.end = end;
            this.piece = end - start; // total length that this thread handles on the sequence, used to identify the
                                      // barrier location
            switch (granularity) {
                case NONE:
                    piece = 1;
                    break;
                case QUARTER:
                    piece = piece / 4;
                    break;
                case HALF:
                    piece = piece / 2;
                    break;
                case MAX:
                    break;
                default:
                    break;
            }
            // if the length is too small, we reset to base case (granularity = MAX)
            if (piece == 0)
                piece = end - start;
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
         * The Runnable part of the class. Sorts an unordered sequence using bitonic
         * loops. The algorithm stops at specific barriers to wait for other threads to
         * finish the loop at that barrier
         * then moves to the next loop
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

                    // i is the merge element
                    for (int i = start; i < end; i++) {
                        int ixj = i ^ j; // xor: all the bits that are on in one and off in the other

                        // only compare if ixj is to the right of i
                        if (ixj > i) {
                            if ((i & k) == 0 && seq[i] > seq[ixj]) {
                                swap(i, ixj);
                            }
                            if ((i & k) != 0 && seq[i] < seq[ixj]) {
                                swap(i, ixj);
                            }
                        }

                        try {
                            // depending on the granularity, we apply the barrier accordingly
                            // i + 1 because i from 0 to end - 1, while piece is most likely power of 2
                            if ((i + 1) % piece == 0) {
                                barrier.await();
                            }
                        } catch (BrokenBarrierException | InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }

        private double[] seq;
        private CyclicBarrier barrier;
        private int start, end, piece; // end & start are indexes of the original seq that this thread is assigned to
                                       // do the comparators

    }

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        int work = 0;
        while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {
            double[] data = RandomArrayGenerator.getArray(N);
            CyclicBarrier barrier = new CyclicBarrier(N_THREADS);
            Thread[] threads = new Thread[N_THREADS];
            for (int i = 0; i < threads.length; i++) {
                int piece = N / N_THREADS;
                // the first thread starts at 0 til the end of the piece, the 2nd thread starts
                // at the next pieice and so on.
                int startIndex = i * piece;
                // if id is not final thread, then move to the end of piece by adding 1, else
                // end is already at the last element of data
                int endIndex = i != N_THREADS - 1 ? (i + 1) * piece : N;
                threads[i] = new Thread(new BitonicLoopParallel(data, barrier, GRANULARITY, startIndex, endIndex));
                threads[i].start();
            }

            // join to begin validating the array. If we dont join here then the validation
            // will always fail because of the validation process will finish before the
            // threads.
            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
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
