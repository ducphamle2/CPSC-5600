/*
 * Le Duc Pham
 * CPSC 5600, Seattle University
 * This is free and unencumbered software released into the public domain.
 */

import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

/**
 * Bitonic stage of a bitonic sorting network pipeline. This class takes in
 * two ASC sorted arrays as inputs
 * and sorts them using the bitonic sort algorithm.
 */
public class BitonicStage implements Runnable {

    /**
     * Set up a BitonicStage with two SynchronousQueues to read for input and one to
     * write for output. It includes a name to identify the
     * thread name
     *
     * @param firstInput  where to read the unordered input array
     * @param secondInput where to read the unordered input array
     * @param output      where to write the sorted array
     * @param name        name of the thread
     */
    public BitonicStage(SynchronousQueue<double[]> firstInput, SynchronousQueue<double[]> secondInput,
            SynchronousQueue<double[]> output, String name) {
        this.firstInput = firstInput;
        this.secondInput = secondInput;
        this.output = output;
        this.name = name;
    }

    /**
     * Set up a BitonicStage with two SynchronousQueues to read for input and one to
     * write for output.
     *
     * @param firstInput  where to read the unordered input array
     * @param secondInput where to read the unordered input array
     * @param output      where to write the sorted array
     */
    public BitonicStage(SynchronousQueue<double[]> firstInput, SynchronousQueue<double[]> secondInput,
            SynchronousQueue<double[]> output) {
        this.firstInput = firstInput;
        this.secondInput = secondInput;
        this.output = output;
    }

    /**
     * Default constructor of Bitonic stage used by BitonicSequential
     */
    public BitonicStage() {
    }

    /**
     * flip - reverse the position of an array data, meaning that the array elements
     * start at n - 1 to 0
     * 
     * @param data - the original array data
     * @return reversed array data
     */
    private double[] flip(double[] data) {
        double[] flippedData = new double[data.length];
        int index = 0;
        for (int i = data.length - 1; i >= 0; i--) {
            flippedData[index] = data[i]; // first index of new array = last index of original array
            index++;
        }
        return flippedData;
    }

    /**
     * concat - concat two given arrays into one, where the first args is first,
     * second args is second
     * 
     * @param firstHalf  - the first half of the concat, also an array
     * @param secondHalf - the second half of the concat, also an array
     * @return - the concat array combining the two arrays given
     */
    private double[] concat(double[] firstHalf, double[] secondHalf) {
        int firstLength = firstHalf.length;
        int secondLength = secondHalf.length;
        double[] concatArray = new double[firstLength + secondLength];
        System.arraycopy(firstHalf, 0, concatArray, 0, firstLength); // copy the first array into the new concatArray
        System.arraycopy(secondHalf, 0, concatArray, firstLength, secondLength); // append the 2nd array into the new
                                                                                 // concatArray with new length
        return concatArray;
    }

    /**
     * swap - swap two array elements of the instance variable bitonicSeq
     * 
     * @param i - first index to swap
     * @param j - second index to swap
     */
    private void swap(int i, int j) {
        double temp = bitonicSeq[i];
        bitonicSeq[i] = bitonicSeq[j];
        bitonicSeq[j] = temp;
    }

    /**
     * Compares two elements between n/2 index, and depending on the direction to
     * swap
     * 
     * @param start     starting index to swap in the array
     * @param n         - size of the array we are considering
     * @param direction - array's sorting direction
     */
    private void bitonicMerge(int start, int n, Direction direction) {
        if (direction == Direction.UP) {
            // we loop til start + n / 2 because we are comparing two halfs i & i + n / 2
            // when i reaches n / 2 - 1 then i + n / 2 will reach n - 1, which is the end of
            // the array
            for (int i = start; i < start + n / 2; i++) {
                // with UP, we are sorting in ASC, so if bi[i] > bi[i + n/2 ] => we swap
                if (bitonicSeq[i] > bitonicSeq[i + n / 2])
                    swap(i, i + n / 2);
            }
        } else if (direction == Direction.DOWN) {
            for (int i = start; i < start + n / 2; i++) {
                // DOWN sorts in DESC
                if (bitonicSeq[i] < bitonicSeq[i + n / 2])
                    swap(i, i + n / 2);
            }
        }
    }

    /**
     * Recursively merge and sort the bitonic sequence. The result of this function
     * is a sorted array based on the given direction
     * 
     * @param start     start index of the array to sort
     * @param n         size of the array
     * @param direction direction to sort, can be either up or down
     */
    private void bitonicSort(int start, int n, Direction direction) {
        if (n > 1) {
            // split the bitonic sequence into two chunks to change the sequence into a
            // non-increasing / non-decreasing sequence
            bitonicMerge(start, n, direction);
            // then recursively sort both sides. n / 2 means we sort the first half then the
            // second half
            bitonicSort(start, n / 2, direction);
            bitonicSort(start + n / 2, n / 2, direction);
        }
    }

    /**
     * flip second array and concat with 1st to create a new array, which is a
     * bitonic sequence
     * 
     * @param firstHalf  - sorted array in asc
     * @param secondHalf - sorted array in asc
     * @return sorted concat new array
     */
    public double[] process(double[] firstHalf, double[] secondHalf) {
        // we flip the 2nd half because we assume both arrays are sorted in ASC order,
        // after flipping the secondHalf array is sorted in DESC.
        secondHalf = flip(secondHalf);
        // We flip 2nd one then concat to make the new array a bitonic sequence.
        double[] concatArray = concat(firstHalf, secondHalf);

        // initiate bitonic sequence based on the inputs so that we dont have to pass in
        // the bitonic seq into the bitonic sort method
        bitonicSeq = concatArray;

        bitonicSort(0, bitonicSeq.length, Direction.UP);
        return bitonicSeq;
    }

    /**
     * The Runnable part of the class. Polls the input queue and when ready, process
     * (sort)
     * it and then write it to the output queue.
     */
    @Override
    public void run() {
        // initialize the arrays so that they are not null. We will reset these two's
        // pointers when the queues receive new inputs
        double[] firstArray = new double[1];
        double[] secondArray = new double[1];
        while (firstArray != null && secondArray != null) {
            try {
                // wait for inputs coming from the input queues. We should offer inputs from
                // outside.
                // After timeout, poll() will return null
                firstArray = firstInput.poll(timeout * 1000, TimeUnit.MILLISECONDS);
                secondArray = secondInput.poll(timeout * 1000, TimeUnit.MILLISECONDS);
                if (firstArray != null && secondArray != null) {
                    double[] outputArray = process(firstArray, secondArray);
                    output.offer(outputArray, timeout * 1000, TimeUnit.MILLISECONDS);
                } else {
                    System.out.println(getClass().getName() + " " + name + " got null array");
                }
            } catch (InterruptedException e) {
                return;
            }
        }
    }

    private SynchronousQueue<double[]> firstInput, secondInput, output;
    private String name;
    private static final int timeout = 10; // 10 in seconds

    // Bitonic sort direction
    private enum Direction {
        UP,
        DOWN
    }

    private double[] bitonicSeq; // final sorted sequence of the object after sorting. We create this instance
                                 // variable so that we dont have to pass it as an argument when sorting
}
