#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

using namespace std;
using namespace cv;

const string imgDir = "./images";
const string datasetDir = "./trainset";
const int imgWidth = 20;
const int imgHeight = 20;
const int nInstances = 1000;

//Generator params
const int noisePercent = 2; // 0..100
const int nCenterPoints = 4;
const Point2f centers[] = {
		{ 9.0, 9.0 },
		{ 9.0, 10.0 },
		{ 10.0, 9.0 },
		{ 10.0, 10.0 }
};

const int angle = 20; // 0..180
const int scale = 90; // 0..100

const int maxShiftX = 3;
const int maxShiftY = 3;

int main()
{
	pair<string, int> dirInfo[] = {
		{ "3", 8 },
		{ "4", 10 }
	};

	vector<int> compressionParams;
	compressionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
	compressionParams.push_back(98);

	for (int suit = 1; suit < 3; suit++)
	{
		vector<Mat> initImages;

		for (int k = 1; k < dirInfo[suit - 1].second; k++)
		{
			string imgPath = imgDir + "/" + dirInfo[suit - 1].first + "/" + to_string(k) + ".jpg";
			Mat img = imread(imgPath);
			Mat grayImg;
			cvtColor(img, grayImg, CV_BGR2GRAY);
			threshold(grayImg, grayImg, 100, 255, CV_THRESH_BINARY_INV);
			resize(grayImg, grayImg, Size(imgWidth, imgHeight));
			initImages.push_back(grayImg);
		}

		for (int instanceId = 1; instanceId <= nInstances; instanceId++)
		{
			cout << "Generating instance #" << instanceId << endl;
			int initImgIdx = rand() % initImages.size();
			int centerPointIdx = rand() % nCenterPoints;

			Mat genImg = initImages[initImgIdx].clone();

			// adding noise
			for (int y = 0; y < genImg.size().height; y++)
			{
				for (int x = 0; x < genImg.size().width; x++)
				{
					if (rand() % 101 <= noisePercent)
					{
						*genImg.ptr<uchar>(y, x) = rand() % 256;
					}
				}
			}

			// rotating and scalling
			double scaleCoef = 0.01 * (scale + rand() % (101 - scale));
			double angleValue = rand() % (angle + 1);

			if (rand() % 2 > 0) angleValue *= -1.0;

			Mat rotationMatrix2D = getRotationMatrix2D(centers[centerPointIdx], angleValue, scaleCoef);

			warpAffine(genImg, genImg, rotationMatrix2D, genImg.size());
			
			int shiftX = rand() % (maxShiftX + 1);
			int shiftY = rand() % (maxShiftY + 1);

			// shifting
			Point2f srcTri[3];
			Point2f dstTri[3];

			srcTri[0] = Point2f(0, 0);
			srcTri[1] = Point2f(genImg.cols - 1.f, 0);
			srcTri[2] = Point2f(0, genImg.rows - 1.f);

			dstTri[0] = Point2f(0 + shiftX, 0 + shiftY);
			dstTri[1] = Point2f(genImg.cols - 1.f + shiftX, 0 + shiftY);
			dstTri[2] = Point2f(0 + shiftX, genImg.rows - 1.f + shiftY);

			Mat warpMat = getAffineTransform(srcTri, dstTri);

			warpAffine(genImg, genImg, warpMat, genImg.size());


			// openinng
			if (rand() % 2 > 0)
			{
				Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(3, 3));
				morphologyEx(genImg, genImg, CV_MOP_OPEN, kernel);
			}

			//writing image
			string filename = datasetDir + "/" + dirInfo[suit - 1].first + "/" + to_string(instanceId) + ".jpg";
			cvtColor(genImg, genImg, CV_GRAY2BGR);
			try {

				imwrite(filename, genImg, compressionParams);
			}
			catch (runtime_error& ex) {
				fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
				return 1;
			}
		}
	}


	return 0;
}