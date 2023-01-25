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
    public static final int N = 1 << 4; // FIXME: size of the final sorted array (power of two). Change to 1 << 22
    public static final int TIME_ALLOWED = 1; // FIXME: change to 10 seconds
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

            // while (System.currentTimeMillis() < start + TIME_ALLOWED * 1000) {

            System.out.println("foobar");

            Thread threads[] = new Thread[N_THREADS];
            double[][] data = new double[4][];

            SynchronousQueue<double[]> input = new SynchronousQueue<>();
            SynchronousQueue<double[]> output = new SynchronousQueue<>();

            // double[][] data = new double[4][];
            for (int section = 0; section < data.length; section++) {
                // feed the new array generator to sort
                // inputs.add(input);
                // outputs.add(output);
                double[] randomArray = RandomArrayGenerator.getArray(N / 4);
                StageOne stageOne = new StageOne(input, output, "stage " + section);
                threads[section] = new Thread(stageOne);
                threads[section].start();
                input.put(randomArray);
                data[section] = output.take();
            }

            for (int i = 0; i < data.length; i++) {
                // double[] sortedArray = outputs.get(i).poll();
                if (data[i] != null) {
                    System.out.print("index: " + i + " ");
                    for (double val : data[i]) {
                        System.out.println(val + " ");
                    }
                    System.out.println("");
                }
            }
            System.out.println("foobar");

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
            // BitonicStage bitonic = new BitonicStage();
            // double[] penult1 = bitonic.process(data[0], data[1]);
            // double[] penult2 = bitonic.process(data[2], data[3]);
            // double[] ult = bitonic.process(penult1, penult2);
            // if (!RandomArrayGenerator.isSorted(ult) || N != ult.length) {
            // System.out.println("failed");
            // System.exit(1);
            // }
            // work++;
            // System.out.println("start with time allowed: " + (start + TIME_ALLOWED *
            // 1000));
            // System.out.println("current time millis: " + System.currentTimeMillis());
            // System.out.println("should break: " + ((start + TIME_ALLOWED * 1000) <
            // System.currentTimeMillis()));
            // }
            System.out.println("sorted " + work + " arrays (each: " + N + " doubles) in "
                    + TIME_ALLOWED + " seconds");
        } catch (InterruptedException e) {
            System.out.println("error: " + e);
        }
    }
}

// how to stop the threads: throw interrupt to stop the thread from running