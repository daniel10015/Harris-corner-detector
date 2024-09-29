#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "utils.h"

struct Pixel {
    uint8_t r, g, b;
};

class PPMImage 
{
public:
    size_t width = 0;
    size_t height = 0;
    int maxColorValue = 0;
    std::vector<Pixel> pixels;
    PPMImage() = default;
    bool load(const std::string& filename);
    void skipComments(std::ifstream& file);
};

struct GrayscaleImage
{
    size_t width = 0;
    size_t height = 0;
    int maxColorValue = 0;
    std::vector<float> pixels;
};

void ppmToGrayscale(GrayscaleImage* gray, PPMImage* ppm);

// Save greyscale pixel data to a file
// Save grayscale image data in PGM format (Portable Graymap)
void saveGrayscaleImagePGM(const std::vector<float>& pixels, int width, int height, const std::string& filename);

// default radius to 5 if none is given
// TODO make it parallel to increase performance
void gausBlurrGrayscale(GrayscaleImage* gray, int radius = 5);