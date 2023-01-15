public class HW1 {
    private static final int N_THREADS = 8;

    private HW1(int data[], boolean parallel) throws InterruptedException {
        this.data = data;
        if (parallel)
            prefixSumsParallel();
        else
            prefixSumsSequential();
    }

    private class Encoder implements Runnable {
        final int start, end;

        Encoder(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public void run() {
            for (int i = start; i < end; i++)
                data[i] = Codec.encode(data[i]);
        }
    }

    private class Decoder extends Encoder {
        Decoder(int start, int end) {
            super(start, end);
        }

        @Override
        public void run() {
            for (int i = start; i < end; i++)
                data[i] = Codec.decode(data[i]);
        }
    }

    private void prefixSumsParallel() throws InterruptedException {
        Thread threads[] = new Thread[N_THREADS];
        int piece = data.length / N_THREADS;

        int start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            int end = i == N_THREADS - 1 ? data.length : start + piece;
            threads[i] = new Thread(new Encoder(start, end));
            threads[i].start();
            start = end;
        }
        for (int i = 0; i < N_THREADS; i++)
            threads[i].join();

        int encodedSum = 0;
        for (int i = 0; i < data.length; i++) {
            encodedSum += data[i];
            data[i] = encodedSum;
        }

        start = 0;
        for (int i = 0; i < N_THREADS; i++) {
            int end = i == N_THREADS - 1 ? data.length : start + piece;
            threads[i] = new Thread(new Decoder(start, end));
            threads[i].start();
            start = end;
        }
        for (int i = 0; i < N_THREADS; i++)
            threads[i].join();
    }

    private void prefixSumsSequential() {
        int encodedSum = 0;
        for (int i = 0; i < data.length; i++) {
            encodedSum += Codec.encode(data[i]);
            data[i] = Codec.decode(encodedSum);
        }
    }

    private final int data[];

    public static void main(String[] args) throws InterruptedException {
        int length = 1_000_000;
        int data[] = new int[length];
        for (int i = 0; i < length; i++)
            data[i] = 1;
        data[0] = 6;

        // Do it sequentially first
        long then = System.nanoTime();
        HW1 sequential = new HW1(data, false);
        long now = System.nanoTime();

        // System.out.println(Arrays.toString(data));
        System.out.println("[0]: " + data[0]);
        System.out.println("[" + length / 2 + "]: " + data[length / 2]);
        System.out.println("[end]: " + data[length - 1]);

        System.out.println("sequential time: " + (now - then) / 1_000_000 + "ms");

        // Compare to parallel
        for (int i = 0; i < length; i++)
            data[i] = 1;
        data[0] = 6;
        then = System.nanoTime();
        HW1 parallel = new HW1(data, true);
        now = System.nanoTime();

        // System.out.println(Arrays.toString(data));
        System.out.println("[0]: " + data[0]);
        System.out.println("[" + length / 2 + "]: " + data[length / 2]);
        System.out.println("[end]: " + data[length - 1]);

        System.out.println("parallel time: " + (now - then) / 1_000_000
                + "ms with " + N_THREADS + " threads");
    }
}
