#include "Image.h"
#include <iostream>
#include "HarrisOps.cuh"
#include "Timer.h"

int main(int argc, char** argv)
{
	// initially load image from color -> grayscale -> blur (to remove noise)
	// then start compute

	std::string image_path = "images/sage_1.ppm";
	// object to load in ppm image data (both binary and plain text)
	PPMImage img;
	GrayscaleImage grayImg;
	img.load(image_path);
	ppmToGrayscale(&grayImg, &img);
	// save grayscale image
	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size()-4) + "_grayscale.ppm");
	timer time;
	time.start();
	gausBlurrGrayscale(&grayImg, 5);
	std::cout << "CPU time taken for (1) pass: " << NS_TO_S(time.Mark()) << std::endl;
	// save blurred image
	saveGrayscaleImagePGM(grayImg.pixels, grayImg.width, grayImg.height, image_path.substr(0, image_path.size() - 4) + "_grayscale_gaus.ppm");

	// start using CUDA kk
	GrayscaleImage outputImg = {};
	time.start();
	HarrisCornerDetection(outputImg, grayImg);
	std::cout << "GPU time taken for (13) pass: " << NS_TO_S(time.Mark()) << std::endl;

	// save final image
	saveGrayscaleImagePGM(outputImg.pixels, outputImg.width, outputImg.height, image_path.substr(0, image_path.size() - 4) + "_grayscale_R.ppm");

	return 0;
}
