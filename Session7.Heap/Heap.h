//
// Created by Kevin Lundeen on 10/21/20.
// Seattle University, CPSC 5005, Session 7
//

#pragma once

/**
 * This Heap class implements a Priority Queue ADT of integers
 *
 * The priority queue enqueues items in any order, but the dequeue order
 * is determined by the natural ordering of the elements. The item dequeued
 * is always the minimum value of all the items currently in the priority
 * queue.
 *
 * A priority queue does not permit null elements.
 *
 * Note that some priority queue authors would call our 'enqueue' method 'offer'
 * and our 'dequeue' method 'poll'. And Carrano calls them 'add' and 'remove'.
 *
 * A priority queue has the same methods as a queue but the semantics of dequeue
 * (and correspondingly, peek) are different.
 */
class Heap {
public:
    /**
     * Simple constructor.
     */
    Heap();

    /**
     * All-at-once constructor.
     * This is more efficient than enqueueing them all individually.
     * @param data  array of integers to enqueue
     * @param size  number of integers in data
     */
    Heap(const int *data, int size);

    /**
     * Destructor.
     */
    ~Heap();

    /**
     * Copy constructor (deep copy).
     * @param other source of this object's data
     */
    Heap(const Heap &other);

    /**
     * Assignment operator (deep copy).
     * @param rhs source of this object's data
     * @return *this
     */
    Heap &operator=(const Heap &rhs);

    /**
     * Add an element to the queue. Items need not be unique.
     * @param newItem to add to the priority queue
     */
    void enqueue(int newItem);

    /**
     * Remove the least element from the queue.
     * @return the least element
     */
    int dequeue();

    /**
     * Check if there are any elements in the queue.
     * @return true if there are no elements
     */
    bool empty() const;

    /**
     * Fetch the element that would be returned by dequeue.
     * @return the least element
     */
    int peek() const;

    /**
     * Sort the given array in place in ascending order using heapsort,
     * O(n log n).
     *
     * @param data to be sorted
     * @param size number of items in data
     */
    static void heapsort(int *data, int size);

    /**
     * Check if the heap invariants are true for every node.
     *
     * @return true if every node is correct
     */
    bool isValid();

private:
    static const int ROOT = 0;
    static const int INITIAL_CAPACITY = 17;
    int size;
    int capacity;
    int *data;

    /**
     * The value at data[index] may violate the heap invariants by being too low.
     * If so, fix it by swapping with ancestors as necessary.
     *
     * @param index of data that may be too low relative to parent (and further
     *              ancestors)
     */
    void percolateUp(int index);

    /**
     * The value at data[index] may violate the heap invariants by being too
     * high. If so, fix it by swapping with descendants as necessary.
     *
     * @param index of data that may be too high relative to children (and
     *              further descendants)
     */
    void percolateDown(int index);

    /**
     * Construct a heap from arbitrarily-ordered elements in the data array.
     */
    void heapify();

    /**
     * Get the index of the parent of a given node in this heap.
     * Does not check if parent index is the root (parent(0) returns 0).
     *
     * @param childIndex child address
     * @return index of the left child of parentIndex in the data array
     */
    static int parent(int childIndex);

    /**
     * Check if the given node has a left child.
     *
     * @param parentIndex parent address
     * @return true if the left child of parentIndex is a current member of the
     *         heap
     */
    bool hasLeft(int parentIndex) const;

    /**
     * Check if the given node has a right child.
     *
     * @param parentIndex parent address
     * @return true if the right child of parentIndex is a current member of
     *         the heap
     */
    bool hasRight(int parentIndex) const;

    /**
     * Get the index of the left child of a given node in this heap.
     * Does not check if the child is a current member of this heap.
     *
     * @param parentIndex parent address
     * @return index of the left child of parentIndex in the data array
     */
    static int left(int parentIndex);

    /**
     * Get the index of the right child of a given node in this heap.
     * Does not check if the child is a current member of this heap.
     *
     * @param parentIndex parent address
     * @return index of the right child of parentIndex in the data array
     */
    static int right(int parentIndex);
};
