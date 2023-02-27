/**
 * @file Mnist.h - a class to to hold Mnist dataset & calculate euclid distance between two images
 * @author Le Duc Pham
 * @see "Seattle University, CPSC5600, Winter 2023"
 */
#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <array>

class Mnist
{
public:
    std::array<u_char, 784> image; // assume that each image has 28 * 28 dimensions

    // ctor's and related
    Mnist(); // not "explicit" since we want automatic conversion from array
    Mnist(std::array<u_char, 784> as_image);
    explicit Mnist(u_char *image);

    // distance, etc.
    double euclidDistance(const Mnist &other) const;

    // read images & labels from the mnist dataset & set them to the data & image labels pointer
    static void images(Mnist **data, u_char **imageLabels, int *size, int *dimension);
};
