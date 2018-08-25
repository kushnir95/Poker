#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Features_Mapper.h"

using namespace std;
using namespace cv;


/*
���� Features_Mapper ���������� ������ ���������� � ���� float ������� �������.
public ��������� � ���� prepare_descriptors() � ���� ��������.
����� ������� � ����������.
prepare_descriptors(string datasetDirectory, int n) - ���������� ���� � n ��������� (�� ������ �� datasetDirectory) � �����������.
�� ����� ���� ������� descriptorNumber x n.

prepare_descriptors(Mat image) - ���������� ������ ���������� � �����������.
�� ����� ���� ������� descriptorNumber x 1. (���� �� ���������������)

������� ����� ��������, ���� �������� ���� ����� �� �����.
*/

Features_Mapper::Features_Mapper() {}

vector<Point> Features_Mapper::prepare_contour(Mat & image)
{
	normalize(image, image, 0, 255, NORM_MINMAX);
	threshold(image, image, 150, 255, THRESH_BINARY_INV);
	vector<vector<Point>> contours;
	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int cArea = 0;
	int maxArea = 0;
	int maxArea_idx = 0;
	if (contours.size() != 1)
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> c = contours.at(j);
			cArea = contourArea(c);
			if (cArea > maxArea) {
				maxArea = cArea;
				maxArea_idx = j;
			}
					
		}
	return contours.at(maxArea_idx);
}

double Features_Mapper::calculateCoef(vector<Point>& c)
{
	vector<Point> hull;
	convexHull(c, hull);
	return contourArea(hull) / contourArea(c);
}

vector<vector<Point>> Features_Mapper::split_contour(vector<Point>& c)
{
	int minY = INT_MAX;
	int maxY = 0;
	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.y < minY) minY = p.y;
		if (p.y > maxY) maxY = p.y;
	}
	int meanY = (maxY + minY) / 2;
	vector<Point> top, bottom;
	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.y < meanY) top.push_back(p);
		else bottom.push_back(p);
	}
	return { top, bottom };
}

Mat Features_Mapper::prepare_descriptors(string datasetDirectory, int n){
	Mat trainSet = Mat::zeros(Size(4, n), CV_32FC1);		// ����������� ������� ���� 4 � n (descriptorNumber = 4) ���� FLOAT32
	for (int i = 0; i < n; i++) {							// ���� ��������� �� n ����������� � �������� ��������.
		Mat image = imread(datasetDirectory + "/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE);	// ��������� ���������� � GRAYSCALE
		vector<Point> pattern_contour = prepare_contour(image);		// �� ���������� ���������� ��������� ������
		vector<vector<Point>> splited = split_contour(pattern_contour);		// ������ ����������� �� ������ � ����� ���������.

		trainSet.at<float>(i, 0) = calculateCoef(pattern_contour);		// ������������� �������� coef = contourArea(hull) / contourArea(c)
		trainSet.at<float>(i, 1) = calculateCoef(splited.at(0));		// ��� ������� � ����� ������� � ���������� � � = {0, 1, 2}
		trainSet.at<float>(i, 2) = calculateCoef(splited.at(1));		// ��� ������� ���������� (������)

		RotatedRect rect = minAreaRect(pattern_contour);					// ����������� �����������, ���� �������� ���� ������� ������,
		double c_coef1 = rect.size.area() / contourArea(pattern_contour);	// ��� ������ (��������) �������. ������������� ��������
																			// coef = rect.size.area() / contourArea(pattern_contour)
		trainSet.at<float>(i, 3) = c_coef1;									// � ���������� � ��������� � = 3 ������� ������.

	}
	return trainSet;
}

Mat Features_Mapper::prepare_descriptors(Mat image){
	Mat m = Mat::zeros(Size(4, 1), CV_32FC1);

	vector<Point> pattern_contour = prepare_contour(image);
	vector<vector<Point>> splited = split_contour(pattern_contour);

	m.at<float>(0, 0) = calculateCoef(pattern_contour);
	m.at<float>(0, 1) = calculateCoef(splited.at(0));
	m.at<float>(0, 2) = calculateCoef(splited.at(1));

	RotatedRect rect = minAreaRect(pattern_contour);
	double c_coef1 = rect.size.area() / contourArea(pattern_contour);

	m.at<float>(0, 3) = c_coef1;

	return m;
}

Features_Mapper::~Features_Mapper() {}
