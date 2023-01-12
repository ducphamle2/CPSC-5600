/**
 * Based on idea from Matthew Flatt, Univ of Utah
 */
#include <iostream>
#include "ThreadGroup.h"
using namespace std;

const int N_THREADS = 2;

struct ShareData
{
	int *data;
	int length;
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
 * @class Example - the template argument to ThreadGroup just has
 * to have an implementation of the ()-operator. (If you're
 * familiar with the STL, this is how functionality is passed in
 * to hash tables, etc.).
 */
class EncodeThread
{
public:
	void operator()(int id, void *sharedData)
	{
		ShareData *ourData = (ShareData *)sharedData;
		int piece = ourData->length / N_THREADS;
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
 * @class Example - the template argument to ThreadGroup just has
 * to have an implementation of the ()-operator. (If you're
 * familiar with the STL, this is how functionality is passed in
 * to hash tables, etc.).
 */
class DecodeThread
{
public:
	void operator()(int id, void *sharedData)
	{
		ShareData *ourData = (ShareData *)sharedData;
		int piece = ourData->length / N_THREADS;
		int start = id * piece;
		// if id is not final thread, then move to the end of piece by adding 1, else end is already at the last element of data
		int end = id != N_THREADS - 1 ? (id + 1) * piece : ourData->length;
		for (int i = start; i < end; i++)
		{
			// decode the prefix sum and put it back into the data array
			ourData->data[i] = decode(ourData->data[i]);
		}
	}
};

void prefixSums(int *data, int length)
{
	ShareData ourData;
	ourData.data = data;
	ourData.length = length;
	int encodedSum = 0;

	// for (int i = 0; i < length; i++)
	// {
	// 	cout << "data element: " << ourData.data[i] << endl;
	// }

	ThreadGroup<EncodeThread> encoders;
	for (int t = 0; t < N_THREADS; t++)
	{
		encoders.createThread(t, &ourData);
	}
	encoders.waitForAll();

	for (int i = 0; i < length; i++)
	{
		encodedSum += ourData.data[i];
		ourData.data[i] = encodedSum;
	}

	ThreadGroup<DecodeThread> decoders;
	for (int t = 0; t < N_THREADS; t++)
	{
		decoders.createThread(t, &ourData);
	}
	decoders.waitForAll();
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
