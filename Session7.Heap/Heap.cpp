//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#include <utility>
#include <stdexcept>
#include "Heap.h"

using namespace std;

Heap::Heap() {
    size = 0;
    capacity = INITIAL_CAPACITY;
    data = new int[capacity];
}

Heap::~Heap() {
    delete[] data;
}

Heap::Heap(const int *data, int size) {
    this->size = size;
    this->capacity = size;
    this->data = new int[capacity];
    for (int i = 0; i < size; i++)
        this->data[i] = data[i];
    heapify();
}

Heap::Heap(const Heap &other) {
    size = other.size;
    capacity = other.capacity;
    data = new int[capacity];
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Heap & Heap::operator=(const Heap &rhs) {
    if (&rhs != this) {
        delete[] data;
        size = rhs.size;
        capacity = rhs.capacity;
        data = new int[capacity];
        for (int i = 0; i < size; i++)
            data[i] = rhs.data[i];
    }
    return *this;
}

void Heap::enqueue(int newItem) {
    if (size == capacity) {
        capacity *= 2;
        int *oldData = data;
        data = new int[capacity];
        for (int i = 0; i < size; i++)
            data[i] = oldData[i];
    }
    data[size++] = newItem;
    percolateUp(size - 1);
}

int Heap::dequeue() {
    int ret = peek();

    // get last val in heap, copy value to index 0 and decrease size
    data[0] = data[--size];
    // NOTE: add this to sort in place: data[size] = ret;

    // create a recursive helper, percolateDown,
    // that allows you move the removed val
    // in the right place
    percolateDown(ROOT);
    return ret;
}

bool Heap::empty() const {
    return size == 0;
}

int Heap::peek() const {
    if (empty())
        throw invalid_argument("empty queue");
    return data[0];
}

void Heap::heapify() {
    // starting at last parent, work backwards to root, causing every subtree
    // to be made into a heap
    for (int index = size / 2; index >= 0; index--)
        percolateDown(index);
}

void Heap::percolateUp(int index) {
    if (index > 0) {
        int p = parent(index);
        // if in violation of invariants, swap it up
        if (data[p] > data[index]) {
            swap(data[p], data[index]);
            percolateUp(p);
        }
    }
}

void Heap::percolateDown(int index) {
    if (hasLeft(index)) {
        // get minimum of the one or two children
        int child = left(index);
        if (hasRight(index)) {
            int r = right(index);
            if (data[r] < data[child])
                child = r;
        }
        // if in violation of invariants, swap it down
        if (data[child] < data[index]) {
            swap(data[index], data[child]);
            percolateDown(child);
        }
    }
}

bool Heap::hasLeft(int parentIndex) const {
    return left(parentIndex) < size;
}

bool Heap::hasRight(int parentIndex) const {
    return right(parentIndex) < size;
}

int Heap::parent(int childIndex) {
    return (childIndex - 1) / 2;
}

int Heap::left(int parentIndex) {
    return parentIndex * 2 + 1;
}

int Heap::right(int parentIndex) {
    return left(parentIndex) + 1;
}

void Heap::heapsort(int *data, int size) {
    Heap heap(data, size);
    // dequeueing everything will get it in ascending order
    for (int i = 0; i < size; i++)
        data[i] = heap.dequeue();
}

bool Heap::isValid() {
    for (int i = size - 1; i > 0; i--) {
        if (data[parent(i)] > data[i])
            return false;
    }
    return true;
}
