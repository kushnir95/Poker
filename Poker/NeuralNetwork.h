#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class NeuralNetwork: public cv::ml::ANN_MLP
{

public:
	NeuralNetwork();
	void train(const std::vector<cv::Mat>& samples, const std::vector<unsigned int>& labels);
	~NeuralNetwork();
};

