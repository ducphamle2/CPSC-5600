import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

/*
 * Le Duc Pham
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

/**
 * @class BitonicPipeline class per cpsc5600 hw3 specification.
 * @versioon 24-Jan-2020
 */
public class BitonicPipeline {
    public static final int N = 1 << 22; // size of the final sorted array (power of two)
    public static final int TIME_ALLOWED = 10; // seconds
    public static final int TIMEOUT = 10; // 10 seconds
    public static final int N_THREADS = 7; // number of threads for bitonic pipeline
    public static final int N_RANDOM_GEN_THREADS = 4; // number of random generator threads

    /**
     * Main entry for HW3 assignment.
     *
     * @param args not used
     */
    public static void main(String[] args) throws InterruptedException {
        try {
            long start = System.currentTimeMillis();
            int work = 0;

            // initialize arrays of threads so that we can interrupt them later, preventing
            // from spawning too many threads at once.
            Thread threads[] = new Thread[N_THREADS];
            Thread randomThreads[] = new Thread[N_RANDOM_GEN_THREADS];

            SynchronousQueue<double[]> input = new SynchronousQueue<>(); // input queue is output of random gen
                                                                         // thread, also input of the StageOne

            SynchronousQueue<double[]> output = new SynchronousQueue<>(); // output of StageOne, which is input for
                                                                          // one of the two input queues for first
                                                                          // bitonic stage
            // two inputs of bitonic stage
            SynchronousQueue<double[]> firstInput = new SynchronousQueue<>();
            SynchronousQueue<double[]> secondInput = new SynchronousQueue<>();
            SynchronousQueue<double[]> secondStageOutput = new SynchronousQueue<>(); // output of first stage
                                                                                     // bitonic, used for two inputs
                                                                                     // of final stage

            // two inputs of the final bitonic stage
            SynchronousQueue<double[]> finalFirstInput = new SynchronousQueue<>();
            SynchronousQueue<double[]> finalSecondInput = new SynchronousQueue<>();
            SynchronousQueue<double[]> finalOutput = new SynchronousQueue<>(); // final output queue, which is
                                                                               // sorted array

            // initialize and start all threads to wait for inputs in the input queues
            for (int section = 0; section < N_THREADS; section++) {
                // 4 stage one threads
                if (section < 4) {
                    StageOne stageOne = new StageOne(input, output, "stage ONE thread " + section);
                    threads[section] = new Thread(stageOne);
                    threads[section].start();

                } else {
                    // init second stage with two input queues
                    BitonicStage bitonic = new BitonicStage(firstInput, secondInput, secondStageOutput,
                            "stage TWO thread " + section);

                    // if its the final stage then we re-init it
                    if (section == 6) {
                        bitonic = new BitonicStage(finalFirstInput, finalSecondInput, finalOutput,
                                "Final bitonic sort thread");
                    }
                    threads[section] = new Thread(bitonic);
                    threads[section].start();
                }
            }

            while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {

                double[] ult = new double[1]; // placeholder for the final sorted array

                for (int i = 0; i < N_RANDOM_GEN_THREADS; i++) {
                    // initialize $ random generator threads
                    RandomArrayGenerator randomGenerator = new RandomArrayGenerator(N / 4, input);
                    randomThreads[i] = new Thread(randomGenerator);
                    randomThreads[i].start();
                }

                // now we handle the logic after starting all the threads
                // we loop through the number of threads to offer inputs & collect outputs
                // respectively based on the thread number
                for (int section = 0; section < N_THREADS; section++) {
                    if (section < 4) {
                        // identify which input is available. Assume that even section is for first
                        // input, and odd is for second input
                        if (section % 2 == 0) {
                            // output of first stage is put in input of 2nd stage
                            // we can use put() and take() here, but if there's a bug then the app will be
                            // blocked indefinitely.
                            // its best to have timeout using offer & poll so the program can exit
                            firstInput.offer(output.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        } else
                            secondInput.offer(output.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);

                    } else if (section == 6)
                        ult = finalOutput.poll(TIMEOUT, TimeUnit.SECONDS);
                    else {
                        // put data into the queue
                        if (section % 2 == 0)
                            finalFirstInput.offer(secondStageOutput.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        else
                            finalSecondInput.offer(secondStageOutput.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);

                    }

                    // interrupt all random threads to kill them all & create new ones to get new
                    // randoms next round
                    for (int i = 0; i < N_RANDOM_GEN_THREADS; i++) {
                        randomThreads[i].interrupt();
                    }
                }

                if (!RandomArrayGenerator.isSorted(ult) || N != ult.length) {
                    System.out.println("failed");
                    System.exit(1);
                }
                work++;
            }
            System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in "
                    + TIME_ALLOWED + " seconds");

            // interrupt all threads to kill them all
            for (int i = 0; i < N_THREADS; i++) {
                threads[i].interrupt();
                if (i < N_RANDOM_GEN_THREADS) {
                    // also interrupt all random threads
                    randomThreads[i].interrupt();
                }
            }
        } catch (

        InterruptedException e) {
            System.out.println("error: " + e);
        }
    }
}

// how to stop the threads: throw interrupt to stop the thread from running