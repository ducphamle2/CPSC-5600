import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

/*
 * Kevin Lundeen
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

/**
 * @class BitonicPipeline class per cpsc5600 hw3 specification.
 * @versioon 24-Jan-2020
 */
public class BitonicPipeline {
    public static final int N = 1 << 22;
    public static final int TIME_ALLOWED = 10; // seconds
    public static final int TIMEOUT = 10; // 10 seconds
    public static final int N_THREADS = 7;
    public static final int N_RANDOM_GEN_THREADS = 4;

    /**
     * Main entry for HW3 assignment.
     *
     * @param args not used
     */
    public static void main(String[] args) throws InterruptedException {
        try {
            long start = System.currentTimeMillis();
            int work = 0;

            while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {

                // System.out.println("foobar");

                Thread threads[] = new Thread[N_THREADS];
                Thread randomThreads[] = new Thread[N_RANDOM_GEN_THREADS];
                double[] ult = new double[1];

                SynchronousQueue<double[]> input = new SynchronousQueue<>();
                SynchronousQueue<double[]> output = new SynchronousQueue<>();
                SynchronousQueue<double[]> firstInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> secondInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> secondStageOutput = new SynchronousQueue<>();
                SynchronousQueue<double[]> finalFirstInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> finalSecondInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> finalOutput = new SynchronousQueue<>();

                // initialize and start all threads to wait for inputs in the input queues
                for (int section = 0; section < N_THREADS; section++) {
                    if (section < 4) {
                        StageOne stageOne = new StageOne(input, output, "stage ONE thread " + section);
                        threads[section] = new Thread(stageOne);
                        threads[section].start();

                        // initiate random generator threads
                        RandomArrayGenerator randomGenerator = new RandomArrayGenerator(N / 4, input);
                        randomThreads[section] = new Thread(randomGenerator);
                        randomThreads[section].start();

                    } else {
                        BitonicStage bitonic = new BitonicStage(firstInput, secondInput, secondStageOutput,
                                "stage TWO thread " + section);
                        if (section == 6) {
                            bitonic = new BitonicStage(finalFirstInput, finalSecondInput, finalOutput,
                                    "Final bitonic sort thread");
                        }
                        threads[section] = new Thread(bitonic);
                        threads[section].start();
                    }
                }

                // now we handle the logic after starting all the threads
                for (int section = 0; section < N_THREADS; section++) {
                    if (section < 4) {
                        if (section % 2 == 0) {
                            // output of first stage is put in input of 2nd stage
                            firstInput.offer(output.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        } else {
                            secondInput.offer(output.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        }
                    } else if (section == 6) {
                        ult = finalOutput.poll(TIMEOUT, TimeUnit.SECONDS);
                    } else {
                        // put data into the queue
                        if (section % 2 == 0) {
                            finalFirstInput.offer(secondStageOutput.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        } else {
                            finalSecondInput.offer(secondStageOutput.poll(TIMEOUT, TimeUnit.SECONDS), TIMEOUT,
                                    TimeUnit.SECONDS);
                        }
                    }
                }

                // interrupt all threads to kill them & create new ones in the next loop
                for (int i = 0; i < N_THREADS; i++) {
                    threads[i].interrupt();
                    if (i < 4) {
                        // also interrupt all random threads
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
        } catch (InterruptedException e) {
            System.out.println("error: " + e);
        }
    }
}

// how to stop the threads: throw interrupt to stop the thread from running