//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#include <iostream>
#include "Heap.h"
using namespace std;

string tf(bool cond) {
    return cond ? "true" : "false";
}

void randomTest(int size, int range) {
    Heap heap1;
    Heap heap2;

    // add a bunch of things
    cout << "Empty heap1: " << tf(heap1.empty()) << endl;
    cout << "Empty heap2: " << tf(heap2.empty()) << endl;
    for (int i = 0; i < size; i++) {
        int n = rand() % range;
        heap1.enqueue(n);
        heap2.enqueue(n);
        if (i % 4 == 0)
            heap1.dequeue(); // so we can get some dequeues into the mix
    }
    cout << "Filled 1: " << tf(!heap1.empty()) << endl;
    cout << "Heap1 valid: " << tf(heap1.isValid()) << endl;
    cout << "Filled 2: " << tf(!heap2.empty()) << endl;
    cout << "Heap2 valid: " << tf(heap2.isValid()) << endl;
}

void drain(Heap heap) {
    // take them out (and check peek at the same time)
    int prev = -1;
    while (!heap.empty()) {
        if (heap.peek() < prev) {
            cout << "out of order FAIL!!" << endl;
            return;
        }
        prev = heap.peek();
        if (prev != heap.dequeue()) {
            cout << "peek != dequeue FAIL!!" << endl;
            return;
        }
    }
}

void heapifyTest(int size, int range) {
    int data[size];
    for (int i = 0; i < size; i++)
        data[i] = rand() % range;
    Heap heap(data, size);
    cout << "Heapify test: " << (heap.isValid() ? "valid" : "INVALID") << endl;
    drain(heap);
}

void heapsortTest(int size, int range, bool print) {
    int data[size];
    for (int i = 0; i < size; i++)
        data[i] = rand() % range;
    Heap::heapsort(data, size);
    if (print) {
        cout << "sorted: " << endl;
        for (int i = 0; i < size; i++)
            cout << data[i] << " ";
        cout << endl;
    }
}

int main() {
    srand(time(nullptr));
    randomTest(400, 100);
    heapifyTest(1000, 100);
    heapsortTest(50, 100, true);

    /* FIXME: do timing in C++ like in this Java version
     * FIXME: this would also require resize() with dynamic array
    long start = System.nanoTime();
    for (int n = 5_000; n < 100_000_000; n *= 2) {
        heapsortTest(n, 100, false);
        long end = System.nanoTime();
        System.out.println(n + ": " + (end - start) / 1000_000_000.0);
        start = end;
    }
    */

    return EXIT_SUCCESS;
}
