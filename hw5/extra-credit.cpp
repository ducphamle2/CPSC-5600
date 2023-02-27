/**
 * @file ColorKMeansMPI.cpp - implementation code for ColorKMeansMPI class
 * @author Kevin Lundeen
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "MnistKmeans.h"
#include "mpi.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

const int K = 10;

// main test (k-means clustering of X11 colors)
int main()
{
    Mnist *images;
    u_char *imageLabels;

    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // // Set up k-means
    MnistKmeans<K> kMeans;

    int nImages, nDimensions;
    if (rank == 0)
    {
        // Set up some data using colors from X11
        Mnist::images(&images, &imageLabels, &nImages, &nDimensions);
        // cout << "in here" << endl;
        // cout << "number of images: " << nImages << endl;
        // for (int i = 0; i < nImages; i++)
        // {
        //     printf("image[%d]\n", i);
        //     for (int j = 0; j < nDimensions; j++)
        //     {
        //         printf("%d ", images[i].image[j]);
        //     }
        //     printf("\n");
        // }
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
    cv::Mat imageChunks, finalImagesDisplay;
    std::vector<cv::Mat> listImages;

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

    MPI_Finalize();
    return 0;
}