import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;

public class BitonicStage implements Runnable {

    public BitonicStage(SynchronousQueue<double[]> firstInput, SynchronousQueue<double[]> secondInput,
            SynchronousQueue<double[]> output, String name) {
        this.firstInput = firstInput;
        this.secondInput = secondInput;
        this.output = output;
        this.name = name;
    }

    public BitonicStage(SynchronousQueue<double[]> firstInput, SynchronousQueue<double[]> secondInput,
            SynchronousQueue<double[]> output) {
        this.firstInput = firstInput;
        this.secondInput = secondInput;
        this.output = output;
    }

    public BitonicStage() {
    }

    private enum Direction {
        UP,
        DOWN
    }

    private double[] bitonicSeq;

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

    private void bitonicMerge(int start, int n, Direction direction) {
        // System.out.println("start: " + start + " end: " + (start + n / 2));
        for (int i = start; i < start + n / 2; i++) {
            // System.out.println("bitonic merge chunk: " + bitonicSeq[i]);
        }
        if (direction == Direction.UP) {
            for (int i = start; i < start + n / 2; i++) {
                if (bitonicSeq[i] > bitonicSeq[i + n / 2])
                    // swap
                    swap(i, i + n / 2);
            }
        } else if (direction == Direction.DOWN) {
            for (int i = start; i < start + n / 2; i++) {
                if (bitonicSeq[i] < bitonicSeq[i + n / 2])
                    // swap
                    swap(i, i + n / 2);
            }
        }
        // System.out.println("end chunk");
    }

    private void bitonicSort(int start, int n, Direction direction) {
        if (n > 1) {
            bitonicMerge(start, n, direction);
            bitonicSort(start, n / 2, direction);
            bitonicSort(start + n / 2, n / 2, direction);
        }
    }

    private boolean isInArray(int[] array, int value) {
        for (int i = 0; i < array.length; i++) {
            if (value == array[i]) {
                return true;
            }
        }
        return false;
    }

    private void bitonicSortNoRecursion(int start, int n) {
        // UP - i holds the dot
        // DOWN - i + j holds the dot
        Direction dot = Direction.UP;
        int moveICount = 0;
        int movePlusJCount = 0;

        for (int k = 1; k < n; k *= 2) {
            // System.out.println("k: " + k);
            for (int j = k; j >= 1; j /= 2) {
                // when j picks the next value, we come back to step 2, reset the dot to i &
                // move counts
                dot = Direction.UP;
                moveICount = 0;
                movePlusJCount = 0;
                // size of datum is j+1 because we only need to store in total j datums before i
                // reaches j. After that, we can rotate j to keep storing new j index
                int[] datumPlusJ = new int[j + 1];
                int datumIndex = 0;
                for (int i = 0; i < n; i++) {

                    int plusJ = i + j;
                    // System.out.println("index i: " + i + " index i + j: " + plusJ);
                    if (plusJ > n - 1) {
                        // if plusJ cant move then we break the loop, j picks the new value
                        break;
                    }
                    // System.out.println("move i count: " + moveICount);
                    // System.out.println("move plus j count: " + movePlusJCount);
                    // based on bitonic dance, if total move of i >= k*2 => switch the dot
                    if (moveICount >= k * 2) {
                        dot = Direction.DOWN;
                        moveICount = 0;
                    }
                    if (movePlusJCount >= k * 2) {
                        dot = Direction.UP;
                        movePlusJCount = 0;
                    }
                    // System.out.println("dot after checking move count: " + dot);
                    // we increment the move of i and i+j based on the dot status to measure the
                    // current move count.
                    switch (dot) {
                        case UP:
                            moveICount++;
                            break;
                        case DOWN:
                            movePlusJCount++;
                        default:
                            break;
                    }
                    // if the current index i has been compared before aka in the datum list, then
                    // we move left instead of comparing
                    if (i != 0 && isInArray(datumPlusJ, i)) {
                        continue;
                    }
                    // store the datum that we have compared so that we can search the datum in the
                    // next iteration
                    datumPlusJ[datumIndex] = plusJ;
                    datumIndex++;
                    // rotate datum list so we can store the datum list efficiently & fast lookup
                    if (datumIndex == datumPlusJ.length)
                        datumIndex = 0;

                    // System.out.println("bitonic elements to compare: " + bitonicSeq[i] + " and :"
                    // + bitonicSeq[plusJ]);

                    // now we compare based on the dots
                    switch (dot) {
                        // i holds the dot
                        case UP:
                            if (bitonicSeq[i] > bitonicSeq[plusJ]) {
                                // System.out.println("swap up");
                                swap(i, plusJ);
                            }
                            break;
                        // i + j holds the dot
                        case DOWN:
                            if (bitonicSeq[plusJ] > bitonicSeq[i]) {
                                // System.out.println("swap down");
                                swap(i, plusJ);
                            }
                        default:
                            break;
                    }
                }
            }
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

        // for (int i = 0; i < bitonicSeq.length; i++) {
        // System.out.println("bitonic sequence before sorting: " + bitonicSeq[i]);
        // }

        // bitonicSortNoRecursion(0, bitonicSeq.length);
        bitonicSort(0, bitonicSeq.length, Direction.UP);

        // for (int i = 0; i < bitonicSeq.length; i++) {
        // System.out.println("bitonic sequence after sorting: " + bitonicSeq[i]);
        // }
        // System.out.println("end process");
        return bitonicSeq;
    }

    /**
     * The Runnable part of the class. Polls the input queue and when ready, process
     * (sort)
     * it and then write it to the output queue.
     */
    @Override
    public void run() {
        double[] firstArray = new double[1];
        double[] secondArray = new double[1];
        while (firstArray != null && secondArray != null) {
            try {
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
}
