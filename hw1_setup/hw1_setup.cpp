/**
 * Based on idea from Matthew Flatt, Univ of Utah
 */
#include <iostream>
#include "ThreadGroup.h"
using namespace std;

const int N_THREADS = 2; /// Number of threads used to encode and decode data elements

/**
 * @struct ShareData - data structure containing the original array data & its length
 *
 * Because we already know the size of data, using ShareData allows us to group the data array and its length together without re-calculating the size of the data array
 * every time we encode and decode the elements.
 */
struct ShareData
{
	int *data;	/// original data array to calculate prefix sum
	int length; /// the size of the data array

	/**
	 * ShareData - constructor
	 *
	 * @param data - original data array
	 * @param length - size of the data array
	 */
	ShareData(int *data, int length)
		: data(data), length(length)
	{
	}
};

int encode(int v)
{
	// do something time-consuming (and arbitrary)
	for (int i = 0; i < 500; i++)
		v = ((v * v) + v) % 10;
	return v;
}

int decode(int v)
{
	// do something time-consuming (and arbitrary)
	return encode(v);
}

/**
 * @class EncodeThread - A concrete type of the ThreadGroup class, which overloads the ()-operator method to encode the data elements
 */
class EncodeThread
{
public:
	/**
	 * ()-operator - a function that is called when we start the thread
	 *
	 * @param id - thread id number
	 * @param sharedData - arbitrary data. Its data type is void * to disable compiler type checking. In our case, this should be ShareData struct
	 */
	void operator()(int id, void *sharedData)
	{
		ShareData *ourData = (ShareData *)sharedData;
		// each thread handles a piece of the data array
		int piece = ourData->length / N_THREADS;
		// the first thread starts at 0 til the end of the piece, the 2nd thread starts at the next pieice and so on.
		int start = id * piece;
		// if id is not final thread, then move to the end of piece by adding 1, else end is already at the last element of data
		int end = id != N_THREADS - 1 ? (id + 1) * piece : ourData->length;
		for (int i = start; i < end; i++)
		{
			// we encode all the values in the data array to prepare for accumulation
			ourData->data[i] = encode(ourData->data[i]);
		}
	}
};

/**
 * @class DecodeThread - A concrete type of the ThreadGroup class, which overloads the ()-operator method to decode the data elements
 */
class DecodeThread
{
public:
	/**
	 * ()-operator - a function that is called when we start the thread
	 *
	 * @param id - thread id number
	 * @param sharedData - arbitrary data. Its data type is void * to disable compiler type checking. In our case, this should be ShareData struct
	 */
	void operator()(int id, void *sharedData)
	{
		ShareData *ourData = (ShareData *)sharedData;
		// each thread handles a piece of the data array
		int piece = ourData->length / N_THREADS;
		// the first thread starts at 0 til the end of the piece, the 2nd thread starts at the next pieice and so on.
		int start = id * piece;
		// if id is not final thread, then move to the end of piece by adding 1, else end is already at the last element of data
		int end = id != N_THREADS - 1 ? (id + 1) * piece : ourData->length;
		for (int i = start; i < end; i++)
		{
			// decode the prefix sum and put it back into the data array to finish the algorithm
			ourData->data[i] = decode(ourData->data[i]);
		}
	}
};

/**
 * prefixSums() - function that encodes & calculate the prefix sum all the data elements, then decode these sums
 */
void prefixSums(int *data, int length)
{
	// initialize our shared data
	ShareData *ourData = new ShareData(data, length);
	int encodedSum = 0;

	// Create an EncodeThread ThreadGroup class
	//
	// Any thread created will call the ()-operator method of the EncodeThread class, which will encode the data array elements
	//
	// waitForAll() will wait for all the threads to finish and release memory.
	ThreadGroup<EncodeThread> encoders;
	for (int t = 0; t < N_THREADS; t++)
	{
		encoders.createThread(t, ourData);
	}
	encoders.waitForAll();

	// Do accumulation in sequential and re-assign the data elements to the according sums
	//
	// We cannot do this in parallel because the current sum depends on the previous sum value.
	for (int i = 0; i < length; i++)
	{
		encodedSum += ourData->data[i];
		ourData->data[i] = encodedSum;
	}

	// Create an DecodeThread ThreadGroup class
	//
	// Any thread created will call the ()-operator method of the DecodeThread class, which will decode the data array elements, which now contains all prefix sum values
	//
	// waitForAll() will wait for all the threads to finish and release memory.
	ThreadGroup<DecodeThread> decoders;
	for (int t = 0; t < N_THREADS; t++)
	{
		decoders.createThread(t, ourData);
	}
	decoders.waitForAll();

	// free memory after init ourData
	delete ourData;
}

int main()
{
	int length = 1000 * 1000;

	// make array
	int *data = new int[length];
	for (int i = 1; i < length; i++)
		data[i] = 1;
	data[0] = 6;

	// transform array into converted/deconverted prefix sum of original
	prefixSums(data, length);

	// printed out result is 6, 6, and 2 when data[0] is 6 to start and the rest 1
	cout << "[0]: " << data[0] << endl
		 << "[" << length / 2 << "]: " << data[length / 2] << endl
		 << "[end]: " << data[length - 1] << endl;

	delete[] data;
	return 0;
}
