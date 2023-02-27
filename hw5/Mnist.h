/**
 * @file Mnist.h - a class to hold RGB colors, including an Euclidean distance function, and the X11 color set
 * @author Kevin Lundeen
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
    std::array<u_char, 784> image;

    // ctor's and related
    Mnist(); // not "explicit" since we want automatic conversion from array
    Mnist(std::array<u_char, 784> as_image);
    explicit Mnist(u_char *image, int size);
    friend bool operator==(const Mnist &l, const Mnist &r);

    // distance, etc.
    double euclidDistance(const Mnist &other) const;

    // color set
    static void images(Mnist **data, u_char **imageLabels, int *size, int *dimension);
};
