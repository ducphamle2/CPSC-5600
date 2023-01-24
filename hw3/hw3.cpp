#include <iostream>

const int N = 8;

int main()
{
    for (int k = 1; k < N; k *= 2)
    {
        for (int j = k; j >= 1; j /= 2)
        {
            for (int i = 0; i < N; i++)
            {
                // dot if (smaller)
                // swap
            }
        }
    }
    return 0;
}

// bitonic sort - used to sort a sequence into bitonic sequence
// once we have a bitonic sequence, then we use bitonic merge - to split the sequence down into smaller chunks of bitonic sequence and swap until they are all sorted

// Container - sortable template

// (int m) m - in bitonic recursion is where to start
// std::swap - save memory by assigning pointer?

// j >> 1 is equal j =/ 2, but j >> 1 is much faster
// Exclusive OR: ^

// i & k == 0 - i has the dot
// i & k != 0 - i does not have the dot

// Exam: co the viet bitonic loops k can dung bitwise, ma dung if else bthg cx dc

// parallel dot comparator (each wire in a sub-column)
// Can split in half with two threads to run
// or use even-odd

// with 16 threads, bitonic is 2 times faster than quick sort. Can only 2 times faster because with 1 thread, bitonic is 4-5 times slower than quick sort