public class BitonicStage {

    private enum Direction {
        UP,
        DOWN
    }

    private double[] bitonicSeq;

    private double[] flip(double[] data) {
        double[] flippedData = new double[data.length];
        int index = 0;
        for (int i = data.length - 1; i >= 0; i--) {
            flippedData[index] = data[i];
            index++;
        }
        return flippedData;
    }

    private double[] concat(double[] firstHalf, double[] secondHalf) {
        int firstLength = firstHalf.length;
        int secondLength = secondHalf.length;
        double[] concatArray = new double[firstLength + secondLength];
        System.arraycopy(firstHalf, 0, concatArray, 0, firstLength);
        System.arraycopy(secondHalf, 0, concatArray, firstLength, secondLength);
        return concatArray;
    }

    private void bitonicMerge(int start, int n, Direction direction) {
        System.out.println("start: " + start + " end: " + (start + n / 2));
        for (int i = start; i < start + n / 2; i++) {
            System.out.println("bitonic merge chunk: " + bitonicSeq[i]);
        }
        if (direction == Direction.UP) {
            for (int i = start; i < start + n / 2; i++) {
                if (bitonicSeq[i] > bitonicSeq[i + n / 2]) {
                    // swap
                    double temp = bitonicSeq[i];
                    bitonicSeq[i] = bitonicSeq[i + n / 2];
                    bitonicSeq[i + n / 2] = temp;
                }
            }
        } else if (direction == Direction.DOWN) {
            for (int i = start; i < start + n / 2; i++) {
                if (bitonicSeq[i] < bitonicSeq[i + n / 2]) {
                    // swap
                    double temp = bitonicSeq[i];
                    bitonicSeq[i] = bitonicSeq[i + n / 2];
                    bitonicSeq[i + n / 2] = temp;
                }
            }
        }
        System.out.println("end chunk");
    }

    private void bitonicSort(int start, int n, Direction direction) {
        if (n > 1) {
            bitonicMerge(start, n, direction);
            bitonicSort(start, n / 2, direction);
            bitonicSort(start + n / 2, n / 2, direction);
        }
    }

    public double[] process(double[] firstHalf, double[] secondHalf) {
        secondHalf = flip(secondHalf);
        double[] concatArray = concat(firstHalf, secondHalf);

        // initiate bitonic sequence based on the inputs
        bitonicSeq = concatArray;

        for (int i = 0; i < bitonicSeq.length; i++) {
            System.out.println("bitonic sequence before sorting: " + bitonicSeq[i]);
        }

        bitonicSort(0, bitonicSeq.length, Direction.UP);

        for (int i = 0; i < bitonicSeq.length; i++) {
            System.out.println("bitonic sequence after sorting: " + bitonicSeq[i]);
        }
        System.out.println("end process");
        return bitonicSeq;
    }
}
