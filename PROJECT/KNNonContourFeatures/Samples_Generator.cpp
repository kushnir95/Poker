#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Samples_Generator.h"

using namespace std;
using namespace cv;


/*
Урізана версія генератора. Достатня для обробки контурів.
Бере зображення з imageDirectory, крутить-вертить на 20 градусів і записує в datasetDirectory.
Додатково робить resize по абсолютним величинам, внаслідок чого зображення стискаються-розтягуються.
Це додає гнучкості сітці.
*/
Samples_Generator::Samples_Generator(string imageDirectory, string datasetDirectory, int n, int imgSize)
{
	imgDir = imageDirectory;
	datasetDir = datasetDirectory;
	nInstances = n;

	const int imgWidth = imgSize;
	const int imgHeight = imgSize;
	int center = imgSize/2 + 1;

	const int angle = 20;

	string dirInfo[] = {"SPADE", "CLUB", "HEART", "DIAMOND",
		"TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
		"NINE", "TEN", "JACK", "QUEEN", "KING", "ACE" };

	for (int i = 0; i < 17; i++) {
		vector<Mat> inputImg;
		for (int j = 1; j < 5; j++) {
			Mat input_image = imread(imgDir + "/" + dirInfo[i] + "/" + to_string(j) + ".jpg", IMREAD_GRAYSCALE);
			resize(input_image, input_image, Size(imgWidth, imgHeight));
			inputImg.push_back(input_image);

			for (int k = 0; k < nInstances/4; k++) {
				Mat genImg;
				double angleValue = rand() % (angle + 1);
				if (rand() % 2 > 0) angleValue *= -1.0;

				Mat rotationMatrix2D = getRotationMatrix2D(Point(center, center), angleValue, 1);
				warpAffine(input_image, genImg, rotationMatrix2D, input_image.size());
				imwrite(datasetDir + "/" + dirInfo[i] + "/" + to_string(k + nInstances / 4 * (j-1)) + ".jpg", genImg);
			}
		}
	}
}

Samples_Generator::~Samples_Generator()
{
}
