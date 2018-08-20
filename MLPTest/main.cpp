#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const int width = 20;
const int height = 20;

const int nclasses = 4;
const int ninstances = 8;

const std::string dirPath = "./images/";

Mat trainX, trainY;
vector<int> labels;

void readData()
{
	pair<string, int> dirInfo[] = {
		{"1", 10},
		{"2", 7},
		{"3", 8},
		{"4", 7}
	};

	vector<Mat> images;

	for (int i = 0; i < 4; i++)
	{
		for (int idx = 1; idx <= dirInfo[i].second; idx++)
		{
			string imgPath = dirPath + dirInfo[i].first + "/" + to_string(idx) + ".jpg";
			Mat img = imread(imgPath);
			Mat grayImg;
			cvtColor(img, grayImg, CV_BGR2GRAY);
			threshold(grayImg, grayImg, 100, 255, CV_THRESH_BINARY_INV);
			resize(grayImg, grayImg, Size(width, height));
			images.push_back(grayImg);
			labels.push_back(i);
		}
	}

	trainX = Mat::zeros(Size(width * height, images.size()), CV_32FC1);
	trainY = Mat::zeros(Size(nclasses, images.size()), CV_32FC1);

	int idx = 0;
	for (Mat currentInstance : images)
	{
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				trainX.at<float>(idx, x * width + y) = currentInstance.at<uchar>(y, x);
			}
		}

		trainY.at<float>(idx, labels[idx]) = 1.0;
		idx++;
	}

}

int main()
{
	readData();

	Ptr<cv::ml::ANN_MLP> ANN = cv::ml::ANN_MLP::create();
	Mat_<int> layers(5, 1);
	layers(0) = width * height;
	layers(1) = 50;
	layers(2) = 20;
	layers(3) = 10;
	layers(4) = nclasses;

	ANN->setLayerSizes(layers);
	ANN->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1.0, 0.0);
	ANN->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
	ANN->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

	std::cout << "Start training network..." << std::endl;
	
	ANN->train(trainX, ml::ROW_SAMPLE, trainY);
	
	std::cout << "Finish training network..." << std::endl;

	for (int i = 0; i < labels.size(); i++)
	{
		int pred = ANN->predict(trainX.row(i), noArray());
		int truth = labels[i];
		std::cout << "=============" << i << "===============" << std::endl;
		std::cout << "Predicted: " << pred << "\nTruth: " << truth << std::endl;
	}

	return 0;

}