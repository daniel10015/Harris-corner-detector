#include "Image.h"
#include <iostream>
// images from https://www.cs.cornell.edu/courses/cs664/2003fa/images/

int main(int argc, char** argv)
{
	std::string image_path = "images/car_1.ppm";
	// object to load in ppm image data (both binary and plain text)
	PPMImage img;
	GrayscaleImage grayImg;

	img.load(image_path);

	ppmToGrayscale(&grayImg, &img);

	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size()-4) + "_grayscale.ppm");

	gausBlurrGrayscale(&grayImg, 5);

	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size() - 4) + "_grayscale_gaus.ppm");

	return 0;
}
