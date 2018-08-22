#include "NeuralNetwork.h"
#include <opencv2/opencv.hpp>


NeuralNetwork::NeuralNetwork()
{
}


NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::train(const std::vector<cv::Mat>& samples, const std::vector<unsigned int>& labels)
{
	/*cv::Mat samplesAll = cv::Mat::zeros(cv::Size(samples[0].size().area, samples.size()), CV_32FC1);
	unsigned int classes = std::set<unsigned int>(labels.begin(), labels.end()).size();
	cv::Mat labelsAll = cv::Mat::zeros(cv::Size(samples.size(), classes), CV_32FC1);
*/
}