#pragma once
#include <iostream>


class Features_Mapper
{

private:

	static std::vector<cv::Point> prepare_contour(const cv::Mat& image);

	static double calculateCoef(std::vector<cv::Point>& c);

	static std::vector<std::vector<cv::Point>> split_contour(std::vector<cv::Point>& c, int isHorisontal);

public:

	Features_Mapper();

	static cv::Mat prepare_descriptors(std::string datasetDirectory, int n, int useExtra);

	static cv::Mat prepare_descriptors(cv::Mat image, int useExtra);

	~Features_Mapper();
};
