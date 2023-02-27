/**
 * @file Color.cpp - implementation of Color class methods
 * @author Kevin Lundeen
 * @see "Seattle University, CPSC5600, Winter 2023"
 */

#include "Mnist.h"
#include <utility>
#include <fstream>
#include "ThreadGroup.h"

using namespace std;

const int N = 1 << 10; // this value limit the number of images and labels we want to process. Currently the algorithm can handle a limited number of images (works accurately with N <= 1 << 10. With N larger it seems that the program takes forever to finish)
const int N_THREADS = 4;

struct ShareData
{
    const array<u_char, 784> data; /// original data array to calculate prefix sum
    int length;                    /// the size of the data array
    double distance[N_THREADS];    // list of calculated distances from the threads

    /**
     * ShareData - constructor
     *
     * @param data - original data array
     * @param length - size of the data array
     */
    ShareData(array<u_char, 784> data, int length)
        : data(data), length(length)
    {
        for (int i = 0; i < N_THREADS; i++)
        {
            distance[i] = 0;
        }
    }
};

/**
 * @class DistanceThread - use thread group to calculate distance in parallel
 */
class DistanceThread
{
public:
    /**
     * ()-operator - a function that is called when we start the thread
     *
     * @param id - thread id number
     * @param sharedData - arbitrary data. Its data type is void * to disable compiler type checking. In our case, this should be ShareData struct
     */
    void operator()(int id, void *sharedData, void *other)
    {
        ShareData *ourData = (ShareData *)sharedData;
        ShareData *otherData = (ShareData *)other;
        // each thread handles a piece of the data array
        int piece = ourData->length / N_THREADS;
        // the first thread starts at 0 til the end of the piece, the 2nd thread starts at the next pieice and so on.
        int start = id * piece;
        // if id is not final thread, then move to the end of piece by adding 1, else end is already at the last element of data
        int end = id != N_THREADS - 1 ? (id + 1) * piece : ourData->length;
        double distance = 0.0;
        for (int i = start; i < end; i++)
        {
            // we encode all the values in the data array to prepare for accumulation
            double tmp = ourData->data[i] - otherData->data[i];
            distance += tmp * tmp;
        }
        ourData->distance[id] = distance;
    }
};

Mnist::Mnist()
{
}
/**
 * Mnist constructor taking an array of pixels as the parameter. Each Mnist is an image
 * @param as_image - a mnist image in array of u_char & has fixed size of 784 dimensions
 */
Mnist::Mnist(const array<u_char, 784> as_image) : image(as_image) {}
/**
 * This constructor is used when reading the mnist dataset. I dont know how to cast from u_char* to array so this constructor is needed to convert between them
 */
Mnist::Mnist(u_char *_image)
{
    for (int i = 0; i < int(image.size()); i++)
    {
        image[i] = _image[i];
    }
}

/**
 * Calculate euclid distance of a mnist image
 */
double Mnist::euclidDistance(const Mnist &other) const
{
    double distance = 0;
    int length = int(image.size());
    if (length != int(other.image.size()))
    {
        throw runtime_error("Invalid image size");
    }

    // initialize our shared data
    ShareData *ourData = new ShareData(image, length);
    ShareData *otherData = new ShareData(other.image, length);

    // create a thread group to calculate the euclid distance in parallel. TODO: For some reason, adding threads makes the program run slower than using 1 thread. Why?
    ThreadGroup<DistanceThread> distancesCalculator;
    for (int t = 0; t < N_THREADS; t++)
    {
        distancesCalculator.createThread(t, ourData, otherData);
    }
    distancesCalculator.waitForAll();

    for (int i = 0; i < N_THREADS; i++)
    {
        // printf("distance[%d]: %.2f\n", i, ourData->distance[i]);
        distance += ourData->distance[i];
    }
    delete ourData;
    delete otherData;
    return sqrt(distance);
}

/**
 * this is a helper function to help convert big & small indian, used to unmarshal the mnist dataset
 */
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

/**
 * Read mnist labels and write back to the pointers for later usage
 * @param label_path - the relative or absolute path to the mnist label file
 * @param number_of_labels - placeholder to hold the number of labels of the dataset
 * @param labels - list of labels stored in u_char
 */
void read_mnist_labels(string label_path, int &number_of_labels, u_char *labels)
{
    ifstream labelFile(label_path, std::ios::binary);

    if (labelFile.is_open())
    {
        // each file has a magic number similarly to a checksum. This prevents fake datasets
        int magic_number = 0;
        labelFile.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
            throw runtime_error("Invalid MNIST label file!");

        // read total number of label files
        labelFile.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        // guard check to prevent out of bounds error when N is too large
        int max_size = N;
        if (max_size > number_of_labels)
            max_size = number_of_labels;
        for (int i = 0; i < max_size; i++)
        {
            // read image label
            labelFile.read((char *)&labels[i], 1);
            // string sLabel = std::to_string(int(labels[i]));
        }
    }
    else
    {
        throw runtime_error("Cannot open file `" + label_path + "`!");
    }
}

/**
 * Read mnist images and write back to the pointers for later usage
 * @param image_path - the relative or absolute path to the mnist image file
 * @param number_of_images - placeholder to hold the number of images of the dataset
 * @param n_rows - placeholder for the number of rows an image has
 * @param n_cols - placeholder for the number of columns an image has
 * @param images - list of images stored
 */
void read_mnist_images(string image_path, int &number_of_images, int &n_rows, int &n_cols, Mnist *images)
{

    ifstream imageFile(image_path, std::ios::binary);

    if (imageFile.is_open())
    {
        int magic_number = 0;
        imageFile.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
            throw runtime_error("Invalid MNIST image file!");

        // read total number of image files
        imageFile.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        imageFile.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        imageFile.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        int image_size = n_rows * n_cols;
        int max_size = N;
        if (max_size > number_of_images)
            max_size = number_of_images;
        for (int i = 0; i < max_size; i++)
        {
            // read image pixel
            u_char *image = new u_char[n_rows * n_cols];
            imageFile.read((char *)image, image_size);
            images[i] = Mnist(image);
            delete[] image;
        }
    }
    else
    {
        throw runtime_error("Cannot open file `" + image_path + "`!");
    }
}

void Mnist::images(Mnist **data, u_char **imageLabels, int *size, int *dimension)
{
    int number_of_labels, number_of_images, n_rows, n_cols;
    auto *images = new Mnist[N];
    auto *labels = new u_char[N];
    read_mnist_labels("./t10k-labels-idx1-ubyte", number_of_labels, labels);
    read_mnist_images("./t10k-images-idx3-ubyte", number_of_images, n_rows, n_cols, images);
    *dimension = n_rows * n_cols;
    *data = images;
    *imageLabels = labels;
    *size = N;
}