#pragma once
#include <iostream>


class Features_Mapper
{

private:

	static std::vector<cv::Point> prepare_contour(cv::Mat& image);

	static cv::Mat transform(cv::Mat& image);

	static std::vector<float> functionDivision(cv::Mat& image, int parts, int isVertical);

	static double calculateCoef(std::vector<cv::Point>& c);

	static std::vector<std::vector<cv::Point>> split_contour(std::vector<cv::Point>& c, int isHorisontal);

public:

	Features_Mapper();

	static cv::Mat prepare_descriptors(std::string datasetDirectory, int n, int useExtra);

	static cv::Mat prepare_descriptors(cv::Mat image, int useExtra);

	static cv::Mat prepare_descriptors(cv::Mat image);

	~Features_Mapper();
};
