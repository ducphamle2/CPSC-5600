import java.util.concurrent.SynchronousQueue;

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
    public static final int N = 1 << 22; // FIXME: size of the final sorted array (power of two). Change to 1 << 22
    public static final int TIME_ALLOWED = 10; // FIXME: change to 10 seconds
    public static final int N_THREADS = 7;

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

                double[][] data = new double[4][];

                SynchronousQueue<double[]> input = new SynchronousQueue<>();
                SynchronousQueue<double[]> output = new SynchronousQueue<>();

                // double[][] data = new double[4][];
                for (int section = 0; section < data.length; section++) {
                    // feed the new array generator to sort
                    // inputs.add(input);
                    // outputs.add(output);
                    double[] randomArray = RandomArrayGenerator.getArray(N / 4);
                    StageOne stageOne = new StageOne(input, output, "stage ONE thread " + section);
                    Thread thread = new Thread(stageOne);
                    thread.start();
                    input.put(randomArray);
                    data[section] = output.take();
                    // after collecting the output, we interrupt the thread to stop it from running
                    thread.interrupt();
                }

                // for (int i = 0; i < data.length; i++) {
                // // double[] sortedArray = outputs.get(i).poll();
                // if (data[i] != null) {
                // System.out.print("index: " + i + " ");
                // for (double val : data[i]) {
                // System.out.println(val + " ");
                // }
                // System.out.println("");
                // }
                // }
                // System.out.println("foobar");

                // for (int section = 0; section < data.length; section++) {
                // for (int i = 0; i < data[section].length; i++) {
                // System.out.println("data element with section " + section + ": " +
                // data[section][i]);
                // }
                // }
                // Note that BitonicStage assumes both its input arrays are sorted
                // increasing. It then inverts its second input to form a true bitonic
                // sequence from the concatenation of the first input with the inverted
                // second input.

                SynchronousQueue<double[]> firstInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> secondInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> secondStageOutput = new SynchronousQueue<>();
                double[][] stageTwoOutputs = new double[2][];
                int stageOneIndex = 0;

                for (int i = 0; i < stageTwoOutputs.length; i++) {
                    BitonicStage bitonic = new BitonicStage(firstInput, secondInput, secondStageOutput,
                            "stage TWO thread " + i);
                    Thread thread = new Thread(bitonic);
                    thread.start();
                    // put data into the queue
                    firstInput.put(data[stageOneIndex]);
                    secondInput.put(data[stageOneIndex + 1]);
                    stageTwoOutputs[i] = secondStageOutput.take();
                    thread.interrupt();
                    stageOneIndex += 2;
                }

                SynchronousQueue<double[]> finalFirstInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> finalSecondInput = new SynchronousQueue<>();
                SynchronousQueue<double[]> finalOutput = new SynchronousQueue<>();
                BitonicStage bitonic = new BitonicStage(finalFirstInput, finalSecondInput, finalOutput,
                        "Final bitonic sort thread");
                Thread thread = new Thread(bitonic);
                thread.start();
                finalFirstInput.put(stageTwoOutputs[0]);
                finalSecondInput.put(stageTwoOutputs[1]);
                double[] ult = finalOutput.take();
                thread.interrupt();
                // double[] ult = bitonic.process(stageTwoOutputs[0], stageTwoOutputs[1]);
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