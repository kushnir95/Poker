#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Features_Mapper.h"

using namespace std;
using namespace cv;


/*
Клас Features_Mapper переводить вхідне зображення в набір float значень контура.
public функціями є лише prepare_descriptors() у двох варіантах.
Обидві функції є статичними.
prepare_descriptors(string datasetDirectory, int n) - перетворює серію з n зображень (що лежать по datasetDirectory) в дескриптори.
На виході буде матриця descriptorNumber x n.

prepare_descriptors(Mat image) - перетворює вхідне зображення в дескриптори.
На виході буде матриця descriptorNumber x 1. (Поки не використовується)

Функції майже аналогічні, тому коментар пишу тільки до першої.
*/

Features_Mapper::Features_Mapper() {}

vector<Point> Features_Mapper::prepare_contour(Mat & image)
{
	vector<vector<Point>> contours;
	findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int cArea = 0;
	int maxArea = 0;
	int maxArea_idx = 0;
	if (contours.size() != 1)
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> c = contours.at(j);
			int badPoint = 0;
			for (int k = 0; k < c.size(); k++) {
				Point p = c.at(k);
				if (p.x == 0 || p.x == image.cols || p.y == 0 || p.y == image.rows) {
					badPoint = 1;
				}
			}
			if (!badPoint) {
				int cArea = contourArea(c);
				if (cArea > maxArea) {
					maxArea = cArea;
					maxArea_idx = j;
				}
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

Mat Features_Mapper::transform(Mat& image) {
	Mat img_th;
	//normalize(image, image, 0, 255, NORM_MINMAX);
	threshold(image, img_th, 170, 255, THRESH_BINARY_INV);
	morphologyEx(img_th, img_th, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	vector<Point> c = prepare_contour(img_th);
	RotatedRect rect = minAreaRect(c);
	Point2f pts[4];
	rect.points(pts);

	auto a = norm(pts[0] - pts[1]);
	auto b = norm(pts[1] - pts[2]);

	vector<Point> vertices = {pts[0], pts[1], pts[2], pts[3]};
	
	
	Mat contour_in_mat(vertices);
	contour_in_mat.convertTo(contour_in_mat, CV_32F);

	Mat paper(Size(min(a, b), max(a, b)), CV_8UC1);
	Mat tmp1 = contour_in_mat;
	Mat tmp2;

	if (b >= a)
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << 0, b, a, b, a, 0, 0, 0).reshape(2);
	if (a > b)	
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << b, a, b, 0, 0, 0, 0, a).reshape(2);
	auto p_trans_mat = getPerspectiveTransform(tmp1, tmp2);		
	warpPerspective(img_th, paper, p_trans_mat, paper.size());

	return paper;
}

vector<float> Features_Mapper::functionDivision(Mat& image, int parts, int isVertical) {
	int length = 0;
	vector<float> accumulator(parts);

	if (isVertical) {
		length = image.cols / parts;
		for (int i = 0; i < parts; i++) {
			for (int x = i*length; x < (i+1)*length; x++) {
				for (int y = 0; y < image.rows; y++) {
					if(image.at<uchar>(y, x) == 255)
						accumulator[i] ++;
				}
			}
			accumulator[i] /= length * image.rows;
		}
	}
	else {
		length = image.rows / parts;
		for (int i = 0; i < parts; i++) {
			for (int y = i * length; y < (i + 1)*length; y++) {
				for (int x = 0; x < image.cols; x++) {
					if (image.at<uchar>(y, x) == 255)
						accumulator[i] ++;
				}
			}
			accumulator[i] /= length * image.cols;
		}
	}

	return accumulator;
}

vector<vector<Point>> Features_Mapper::split_contour(vector<Point>& c, int isHorisontal)
{
	if (isHorisontal) {
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
	else {
		int minX = INT_MAX;
		int maxX = 0;
		for (int i = 0; i < c.size(); i++) {
			Point p = c.at(i);
			if (p.x < minX) minX = p.x;
			if (p.x > maxX) maxX = p.x;
		}
		int meanX = (maxX + minX) / 2;
		vector<Point> left, right;
		for (int i = 0; i < c.size(); i++) {
			Point p = c.at(i);
			if (p.x < meanX) left.push_back(p);
			else right.push_back(p);
		}
		return { left, right };
	}
}

Mat Features_Mapper::prepare_descriptors(string datasetDirectory, int n, int parts){
	Mat trainSet = Mat::zeros(Size(parts, n), CV_32FC1);
			
	for (int i = 0; i < n; i++) {							
		Mat image = imread(datasetDirectory + "/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE);	
		
		Mat object = transform(image);
		vector<float> descriptors = functionDivision(object, parts, 1);
		for (int j = 0; j < descriptors.size(); j++) {
			trainSet.at<float>(i, j) = descriptors[j];
		}
		
		/*descriptors = functionDivision(object, parts, 0);
		for (int j = parts; j < parts + descriptors.size(); j++) {
			trainSet.at<float>(i, j) = descriptors[j];
		}*/

	}
	return trainSet;
}

Mat Features_Mapper::prepare_descriptors(Mat image, int parts){
	Mat trainSet = Mat::zeros(Size(parts, 1), CV_32FC1);

	Mat object = transform(image);
	vector<float> descriptors = functionDivision(object, parts, 1);
	for (int j = 0; j < descriptors.size(); j++) {
		trainSet.at<float>(0, j) = descriptors[j];
	}

	/*descriptors = functionDivision(object, parts, 0);
	for (int j = parts; j < parts + descriptors.size(); j++) {
		trainSet.at<float>(0, j) = descriptors[j];
	}*/
	return trainSet;
}

Mat Features_Mapper::prepare_descriptors(Mat image) {
	Mat m = Mat::zeros(Size(5, 1), CV_32FC1);

	vector<Point> pattern_contour = prepare_contour(image);
	vector<vector<Point>> splited = split_contour(pattern_contour, 1);

	m.at<float>(0, 0) = calculateCoef(pattern_contour);
	m.at<float>(0, 1) = calculateCoef(splited.at(0));
	m.at<float>(0, 2) = calculateCoef(splited.at(1));

	RotatedRect rect = minAreaRect(pattern_contour);
	double c_coef1 = rect.size.area() / contourArea(pattern_contour);

	m.at<float>(0, 3) = c_coef1;

	return m;
}

Features_Mapper::~Features_Mapper() {}
