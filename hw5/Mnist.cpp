/**
 * @file Color.cpp - implementation of Color class methods
 * @author Kevin Lundeen
 * @see "Seattle University, CPSC5600, Winter 2023"
 */

#include "Mnist.h"
#include <utility>
#include <fstream>

using namespace std;

const int N = 200;

/**
 * ctor's and related
 */
Mnist::Mnist()
{
}
Mnist::Mnist(const array<u_char, 784> as_image) : image(as_image) {}
Mnist::Mnist(u_char *_image, int size)
{
    for (int i = 0; i < size; i++)
    {
        image[i] = _image[i];
    }
}
bool operator==(const Mnist &l, const Mnist &r)
{
    return l.image == r.image;
}

double Mnist::euclidDistance(const Mnist &other) const
{
    double distance = 0;
    if (int(image.size()) != int(other.image.size()))
    {
        throw runtime_error("Invalid image size");
    }
    for (int i = 0; i < int(image.size()); i++)
    {
        double tmp = image[i] - other.image[i];
        // printf("image[%d]: %d - other.image[%d]: %d\n", i, i, image[i], other.image[i]);
        distance += tmp * tmp;
    }
    return sqrt(distance);
}

int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_labels(string label_path, int &number_of_labels, u_char *labels)
{
    ifstream labelFile(label_path, std::ios::binary);

    if (labelFile.is_open())
    {
        int magic_number = 0;
        labelFile.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
            throw runtime_error("Invalid MNIST label file!");

        // read total number of label files
        labelFile.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);
        for (int i = 0; i < N; i++)
        {
            // read image pixel
            labelFile.read((char *)&labels[i], 1);
            // string sLabel = std::to_string(int(labels[i]));
        }
    }
    else
    {
        throw runtime_error("Cannot open file `" + label_path + "`!");
    }
}

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
        for (int i = 0; i < N; i++)
        {
            // read image pixel
            u_char *image = new u_char[n_rows * n_cols];
            imageFile.read((char *)image, image_size);
            images[i] = Mnist(image, image_size);
        }
    }
    else
    {
        throw runtime_error("Cannot open file `" + image_path + "`!");
    }
}

void Mnist::images(Mnist **data, u_char **imageLabels, int *size, int *dimension)
{
    const int MAX = 200;
    int number_of_labels, number_of_images, n_rows, n_cols;
    auto *images = new Mnist[MAX];
    auto *labels = new u_char[MAX];
    read_mnist_labels("./t10k-labels-idx1-ubyte", number_of_labels, labels);
    read_mnist_images("./t10k-images-idx3-ubyte", number_of_images, n_rows, n_cols, images);
    *dimension = n_rows * n_cols;
    *data = images;
    *imageLabels = labels;
    *size = N;
}