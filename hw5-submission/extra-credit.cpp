/**
 * @file extra-credit.cpp - implementation code to do clustering on the Mnist dataset
 * @author Le Duc Pham
 * @see "Seattle University, CPSC5600, Winter 2023"
 *
 * ANALYSIS
 * This extra credit task is a fun and helpful experience for me.
 * Below is my train of thought to come up with the current version of the program, and my future plan to improve it if I had more time
 *
 * 1. I first read about the MNIST dataset & learned about its data structure. It contains two types of dataset, both of which are in binary: a file for storing images and a file for storing labels.
 * Each dataset has a magic number (the first 4 bytes) which works like a checksum to distinguish itself with other sets
 * The next 4 bytes is the number of images / labels stored in the dataset
 * For the images dataset, the next 8 bytes are the number of rows & columns of an image.
 * And the remaining are pixels for the image dataset, and labels for the label dataset.
 *
 * 2. I spent some time to unmarshal the two test datasets and displayed them on the console (I also used opencv for better visualization). Each image is stored in an array of u_char, and there's an array to store all of these images. This analysis helped me transform the MNIST dataset into a data structure that is similar to the Color data structure, where the only difference is the number of dimensions one has.
 *
 * 3. Next, I integrated the unmarshalling part into our KMeansMPI code by creating two files: Mnist.h & Mnist.cpp (similarly to Color.h & Color.cpp). Since Im not good at C++, I could not find a way to use template to generalize the dimensions of an image. Hence, I just hard-coded 784 as the total number of dimensions of an image (28 rows * 28 columns collected from the dataset). It means that if the image is larger, then the program will not run properly. I also used a constant variable 'N' to control how many images I wanted to process since I wanted to start small for debugging before increasing N.
 *
 * 4. There's a lot of hard-coding parts in Color.cpp because 'rgb' variable only has 3 elements. With the MNIST dataset of 784 elements, I had to generalize that part including calculating the euclid distance. I made an attempt to parallelize the euclid distance calculation by using pthread, but for some reasons the program ran slower than using just one thread.
 *
 * 5. Next, I created MnistKmeans.h to cast the KMeansMPI to use 784 dimensions and extra-credit.cpp which is the entrypoint of the program. I made an attempt to include opencv for the visualization process, but because it is a 3rd party tool, it is now commented so that the graders can run my program at ease.
 *
 * RESULT, REFLECTION.
 *
 * 1. The program runs fine with N less than or equal to 1<<10. If N is increased to 1<<11, the program seems to take forever to finish for some unknown reasons. I have not been able to debug this problem yet. At first, I thought the bottleneck was at the euclid distance function. I then tried to parallelize it using pthread, but the processing time even increased when I pumped up the number of threads.
 *
 * 2. When the number of images increases, the error rate seems to increase. For example, with N = 1<<10, 7 tends to be mistaken with 2, 3 with 8, and there are two clusters with the majority of '1' elements. Maybe using the euclid distance on images is not a good idea for convergence.
 *
 * 3. On the whole, the accuracy rate does not look good when using KMeans on the MNIST dataset.
 *
 * FUTURE IMPROVEMENTS
 *
 * 1. If I had more time, I would look into the bug that caused the program to take so long to finish when increasing N.
 *
 * 2. I also would want to remove the parts where I hard-coded the number 784.
 *
 * 3. Lastly, I would also want to visualize the clusters using a different method that did not depend on a 3rd party library, preferably HTML.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "MnistKmeans.h"
#include "mpi.h"
// #include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>

using namespace std;

const int K = 10; // because there are total 10 possible digits, its best to use K=10

int main()
{
    Mnist *images;
    u_char *imageLabels;

    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Set up k-means
    MnistKmeans<K> kMeans;

    int nImages, nDimensions;
    if (rank == 0)
    {
        // Set up some data from the mnist dataset
        Mnist::images(&images, &imageLabels, &nImages, &nDimensions);
        kMeans.fit(images, nImages);
    }
    else
    {
        kMeans.fitWork(rank);
        MPI_Finalize();
        return 0;
    }

    // get the result
    MnistKmeans<K>::Clusters clusters = kMeans.getClusters();
    // cv::Mat imageChunks, finalImagesDisplay;
    // std::vector<cv::Mat> listImages;

    // Report the result to console
    if (rank == RootProcess)
    {
        int count = 0;
        for (const auto &cluster : clusters)
        {
            printf("cluster[%d]: ", count);
            const auto elements = cluster.elements;
            for (int i = 0; i < int(elements.size()); i++)
            {
                printf("%d ", int(imageLabels[elements[i]]));
            }
            printf("\n");
            count++;

            // Mnist centroid = cluster.centroid;
            // cv::Mat tmp(28, 28, CV_8UC1, &centroid.image);
            // imageChunks.push_back(tmp);
            // int element_size = int(cluster.elements.size());
            // for (int i = 0; i < element_size; i++)
            // {
            //     int elementIndex = cluster.elements[i];
            //     std::array<u_char, 784> element = images[elementIndex].image;
            //     cv::Mat tmp(28, 28, CV_8UC1, &element);
            //     imageChunks.push_back(tmp);
            // }
            // // add dummies so we can display the list with equal rows & cols
            // for (int i = element_size; i < nImages; i++)
            // {
            //     cv::Mat tmp(28, 28, CV_8UC1, cv::Scalar(0, 0, 0));
            //     imageChunks.push_back(tmp);
            // }
            // listImages.push_back(imageChunks);
            // imageChunks.release();
        }
        // cv::hconcat(listImages, finalImagesDisplay);
        // // cv::resize(finalImagesDisplay, finalImagesDisplay, cv::Size(1000, 2000));
        // cv::imshow("label", finalImagesDisplay);
        // while ((cv::waitKey() & 0xEFFFFF) != 27)
        //     ;
    }

    // Also report as a visualization in html for a browser to display
    delete[] images;
    delete[] imageLabels;

    MPI_Finalize();
    return 0;
}