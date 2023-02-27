#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

void read_mnist_data(string image_path, string label_path, int &number_of_labels, int &number_of_images, int &n_rows, int &n_cols)
{
    auto reverseInt = [](int i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream imageFile(image_path, std::ios::binary);
    ifstream labelFile(label_path, std::ios::binary);

    if (imageFile.is_open() && labelFile.is_open())
    {
        int magic_number = 0;
        imageFile.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051)
            throw runtime_error("Invalid MNIST image file!");

        labelFile.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
            throw runtime_error("Invalid MNIST label file!");

        // read total number of label files
        labelFile.read((char *)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        // read total number of image files
        imageFile.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        imageFile.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        imageFile.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        if (number_of_images != number_of_labels)
        {
            throw runtime_error("Number of images and labels are inconsistent!");
        }

        int image_size = n_rows * n_cols;

        u_char *labels = new u_char[number_of_labels];
        u_char *pixels = new u_char[n_rows * n_cols];
        cv::Mat firstImagesChunk, images;
        std::vector<cv::Mat> matrices;

        for (int i = 0; i < 20; i++)
        {
            // read image pixel
            imageFile.read((char *)pixels, image_size);
            labelFile.read((char *)&labels[i], 1);
            string sLabel = std::to_string(int(labels[i]));
            // read label
            // for (int i = 0; i < image_size; i++)
            // {
            //     printf("pixel [%d]: %d\n", i, int(pixels[i]));
            // }

            // if (i == 0)
            // {
            // convert it to cv Mat, and show it
            cv::Mat tmp(n_rows, n_cols, CV_8UC1, pixels);
            firstImagesChunk.push_back(tmp);
            if (i == 9)
            {
                matrices.push_back(firstImagesChunk);
                firstImagesChunk.release();
            }
            // resize bigger for showing
            // }
        }
        matrices.push_back(firstImagesChunk);
        cv::hconcat(matrices, images);
        // cv::resize(images, images, cv::Size(400, 800));
        cv::imshow("label", images);
        cv::waitKey(0);
        delete[] pixels;
        delete[] labels;
    }
    else
    {
        throw runtime_error("Cannot open file `" + image_path + "`!");
    }
}

int main()
{
    int number_of_labels, number_of_images, n_rows, n_cols;
    read_mnist_data("t10k-images-idx3-ubyte", "./t10k-labels-idx1-ubyte", number_of_labels, number_of_images, n_rows, n_cols);
    cout << "number of labels: " << number_of_labels << endl;
    return 0;
}