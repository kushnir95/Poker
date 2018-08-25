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

Mat Features_Mapper::prepare_descriptors(string datasetDirectory, int n, int useExtra){
	Mat trainSet;
	if (!useExtra) {
		trainSet = Mat::zeros(Size(4, n), CV_32FC1);
	} else trainSet = Mat::zeros(Size(5, n), CV_32FC1);
			// створюється матриця нулів 4 х n (descriptorNumber = 4) типу FLOAT32
	for (int i = 0; i < n; i++) {							// Цикл проходить по n зображенням у відповідній директорії.
		Mat image = imread(datasetDirectory + "/" + to_string(i) + ".jpg", IMREAD_GRAYSCALE);	// Зчитується зображення в GRAYSCALE
		vector<Point> pattern_contour = prepare_contour(image);		// Із зображення витягується найбільший контур
		vector<vector<Point>> splited = split_contour(pattern_contour, 1);		// Контур розбивається на верхній і нижній посередині.

		trainSet.at<float>(i, 0) = calculateCoef(pattern_contour);		// Розраховується величина coef = contourArea(hull) / contourArea(c)
		trainSet.at<float>(i, 1) = calculateCoef(splited.at(0));		// для кожного з трьох контурів і записується в х = {0, 1, 2}
		trainSet.at<float>(i, 2) = calculateCoef(splited.at(1));		// для кожного зображення (семпла)

		RotatedRect rect = minAreaRect(pattern_contour);					// Знаходиться прямокутник, який найкраще може описати конутр,
		double c_coef1 = rect.size.area() / contourArea(pattern_contour);	// для цілого (великого) контура. Розраховується величина
																			// coef = rect.size.area() / contourArea(pattern_contour)
		trainSet.at<float>(i, 3) = c_coef1;									// і записується в положення х = 3 кожного семпла.

		if (useExtra) {
			splited = split_contour(pattern_contour, 0);
			trainSet.at<float>(i, 4) = arcLength(splited[0], true)/arcLength(splited[1], true);
		}
		
	}
	return trainSet;
}

Mat Features_Mapper::prepare_descriptors(Mat image, int useExtra){
	Mat m;
	if (!useExtra) {
		m = Mat::zeros(Size(4, 1), CV_32FC1);
	}
	else m = Mat::zeros(Size(5, 1), CV_32FC1);

	vector<Point> pattern_contour = prepare_contour(image);
	vector<vector<Point>> splited = split_contour(pattern_contour, 1);

	m.at<float>(0, 0) = calculateCoef(pattern_contour);
	m.at<float>(0, 1) = calculateCoef(splited.at(0));
	m.at<float>(0, 2) = calculateCoef(splited.at(1));

	RotatedRect rect = minAreaRect(pattern_contour);
	double c_coef1 = rect.size.area() / contourArea(pattern_contour);

	m.at<float>(0, 3) = c_coef1;

	if (useExtra) {
		splited = split_contour(pattern_contour, 0);
		m.at<float>(0, 4) = arcLength(splited[0], true) / arcLength(splited[1], true);
	}

	return m;
}

Features_Mapper::~Features_Mapper() {}
