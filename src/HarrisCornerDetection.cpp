#include "Image.h"
#include <iostream>
#include "HarrisOps.cuh"

int main(int argc, char** argv)
{
	// initially load image from color -> grayscale -> blur (to remove noise)
	// then start compute

	std::string image_path = "images/car_1.ppm";
	// object to load in ppm image data (both binary and plain text)
	PPMImage img;
	GrayscaleImage grayImg;
	img.load(image_path);
	ppmToGrayscale(&grayImg, &img);
	// save grayscale image
	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size()-4) + "_grayscale.ppm");
	gausBlurrGrayscale(&grayImg, 5);
	// save blurred image
	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size() - 4) + "_grayscale_gaus.ppm");

	// start using CUDA kk
	GrayscaleImage outputImg = {};
	HarrisCornerDetection(outputImg, grayImg);

	return 0;
}
