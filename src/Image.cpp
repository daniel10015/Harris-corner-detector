#include "Image.h"
#include <cmath>
#include <algorithm>

void PPMImage::skipComments(std::ifstream& file) {
    std::string line;
    std::cout << "file peeking: " << file.peek() << std::endl;
    // skip newline
    while (file.peek() == '\n')
    {
        std::getline(file, line);  // skip to next line
    }

    while (file.peek() == '#') 
    {
        std::getline(file, line);  // Skip the entire comment line
    }
}

bool PPMImage::load(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Unable to open file " << filename << std::endl;
        return false;
    }

    std::string header;
    file >> header;

    if (header != "P3" && header != "P6") {
        std::cout << "Invalid PPM file type (must be P3 or P6)" << std::endl;
        return false;
    }

    // check whether ppm is plain text or binary
    bool isPlainText = (header == "P3");

    // Skip potential comments before width, height, and maxColorValue
    skipComments(file);

    // Read width, height, and maxColorValue
    file >> width >> height;
    file >> maxColorValue;

    if (maxColorValue != 255) {
        std::cerr << "Unsupported max color value: " << maxColorValue << std::endl;
        return false;
    }

    // Skip the newline after the header
    file.ignore(256, '\n');

    // Allocate pixel data, prepare to write to it from file
    pixels.resize(width * height);

    if (isPlainText) {
        // Load plain text data (P3)
        for (int i = 0; i < width * height; ++i) {
            int r, g, b;
            file >> r >> g >> b;
            pixels[i] = { static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b) };
        }
    }
    else {
        // Load binary data (P6)
        // since these are are bianry (0-255) load in data by writing directly to memory
        file.read((char*)(pixels.data()), width * height * 3);
    }

    return true;
}

// https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
void ppmToGrayscale(GrayscaleImage* gray, PPMImage* ppm)
{
    gray->height = ppm->height;
    gray->width = ppm->width;
    gray->maxColorValue = 255;
    // compute C_linear_i = {R_linear + G_linear + B_linear}_i
    //std::vector<float> linear;
    float linear;
    float grayCol;
    const float R_linear = 0.2126;
    const float G_linear = 0.7152;
    const float B_linear = 0.0722;
    float max = ppm->maxColorValue;
    for (size_t i = 0; i < ppm->pixels.size(); i++)
    {
        linear = R_linear * (ppm->pixels.at(i).r / max) + G_linear * (ppm->pixels.at(i).g / max) + B_linear * (ppm->pixels.at(i).b / max);
        if (linear <= 0.0031308)
        {
            grayCol = 12.92 * linear;
        }
        else
        {
            grayCol = 1.055*(pow(linear, 1/2.4)) - 0.055;
        }
        gray->pixels.push_back(grayCol);
    }
}

// Save grayscale image data in PGM format (Portable Graymap)
void saveGrayscaleImagePGM(const std::vector<float>& pixels, int width, int height, const std::string& filename) 
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // dimensions check
    if (width * height != pixels.size())
    {
        std::cout << "-----WARNING: width/height != pixels-----\n";
        std::cout << "width/height: " << width << "/" << height << ", " << width * height << std::endl;
        std::cout << "pixel size: " << pixels.size() << std::endl;
    }

    // Write PGM header
    file << "P6\r\n" << width << " " << height << "\r\n255\r\n";

    uint8_t byteValue;
    // Write pixel data
    for (float pixel : pixels) {
        byteValue = floatToByte(pixel);
        file.write(reinterpret_cast<char*>(&byteValue), sizeof(byteValue));
        // write 2 more times for *.ppm
        file.write(reinterpret_cast<char*>(&byteValue), sizeof(byteValue));
        file.write(reinterpret_cast<char*>(&byteValue), sizeof(byteValue));
    }

    if (!file.good()) 
    {
        std::cerr << "File write error occurred!" << std::endl;
    }

    file.close();
    std::cout << "Saved " << filename << " successfully as grayscale PPM!" << std::endl;
}

void gausBlurrGrayscale(GrayscaleImage* gray, int radius)
{
    // copy original
    std::vector<float> original = gray->pixels;
    size_t orgWidth = gray->width;
    size_t orgHeight = gray->height;

    // We scale the sigma value in proportion to the radius
    // Setting the minimum standard deviation as a baseline
    double sigma = std::max(double(radius / 2), 1.0);

    // Enforces odd width kernel which ensures a center pixel is always available
    int kernelWidth = (2 * radius) + 1;

    std::vector<float> kernel(kernelWidth*kernelWidth, 0.0);
    float sum = 0;

    // populate kernel 
    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            // compute guas distribution 
            double exponentNumerator = double(-1*(x * x + y * y));
            double exponentDenominator = (2 * sigma * sigma);
            double eExpression = pow(E_APPROX, exponentNumerator / exponentDenominator);
            
            float kernelValue = (eExpression / (2 * PI_APPROX * sigma * sigma));

            // debug
#ifdef _DEBUG
            std::cout << exponentNumerator << std::endl << exponentDenominator 
                << std::endl << eExpression 
                << std::endl << kernelValue << std::endl;
#endif /* DEBUG */
            
            kernel[(y + radius) * kernelWidth + (x + radius)] = kernelValue;
            sum += kernelValue;
        }
    }

    // normalize to sum
#ifdef _DEBUG
// print debug guas matrix
    std::cout << "radius: " << radius << ", sum: " << sum << ", sigma: " << sigma << ", kernel width: " << kernelWidth << std::endl;
    std::cout << "gaus matrix: " << std::endl;
    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            std::cout << kernel[(y + radius) * kernelWidth + (x + radius)] << " | ";
        }
        std::cout << "\n";
    }
#endif /* DEBUG */

    float newSum = 0;
    for (size_t i = 0; i < kernelWidth; i++)
    {
        for (size_t j = 0; j < kernelWidth; j++)
        {
            kernel[(i * kernelWidth + j)] /= sum;
            newSum += kernel[(i * kernelWidth + j)];
        }
    }

    sum = newSum;
    

#ifdef _DEBUG
    // print debug guas matrix
    std::cout << "radius: " << radius << ", sum: " << sum << ", sigma: " << sigma << ", kernel width: " << kernelWidth << std::endl;
    std::cout << "gaus matrix: " << std::endl;
    for (int x = -radius; x <= radius; x++)
    {
        for (int y = -radius; y <= radius; y++)
        {
            std::cout << kernel[(y + radius) * kernelWidth + (x + radius)] << " | ";
        }
        std::cout << "\n";
    }
#endif /* DEBUG */

    // cutoff edges to simplify implementation
    gray->pixels.clear();
    gray->width = orgWidth - (size_t)2 * radius;
    gray->height = orgHeight - (size_t)2 * radius;
    gray->pixels.resize(gray->width * gray->height);

    for (size_t x = radius; x < orgWidth-radius; x++)
    {
        for (size_t y = radius; y < orgHeight-radius; y++)
        {
            float val = 0;

            for (int xKernel = -radius; xKernel <= radius; xKernel++)
            {
                for (int yKernel = -radius; yKernel <= radius; yKernel++)
                {
                   //std::cout << "original: " << original[(y + yKernel) * orgHeight + x + xKernel] << std::endl;
                   //std::cout << original[(y + yKernel) * orgHeight + x + xKernel] * kernel[(yKernel + radius) * kernelWidth + xKernel + radius] << std::endl; 
                    val += original[(y + yKernel) * orgWidth + (x + xKernel)] * kernel[(yKernel + radius) * kernelWidth + (xKernel + radius)];
                }
            }
            //std::cout << "-------value: " << val << std::endl;
            //exit(1);
            gray->pixels[(y - radius) * gray->width + x - radius] = val;
        }
    }

    return;
}