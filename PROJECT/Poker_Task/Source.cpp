#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

string image = "..\\..\\Images\\";

Mat src0, mask;

/* 
���������, �� ���������������� � ������� PreProc().
blurK - ����� ���� ������������� �������.
thresh - ������� ������� ��� ����������.
*/
int blurK = 8;
int thresh = 150;

/*
���������, �� ���������������� � ������� cartContours().
solidity_thresh = 0.9 - ������, �� ��������� ���� �������, � ���� solidity > 0.9 (�� ������ �������).
						solidity - �� ��������� ����� ������� �� ����� convexHull.
area_thresh - ����������� ��������� rec �� ���������� �������� �� ��������.
				rec - �� ��������� ����� ������� �� ���� ���������.
*/
double solidity_thresh = 0.9;
double area_thresh = 1.3;

/*
������� ��� ������������ ����������.
���������� �� 1 ����.
*/
Mat PreProc(Mat src) {
	Mat blured;
	int kernel_size = blurK * 2 + 1;		// ������ ����� ���� �������� ������

	/*
	�������� ���������� �� 640�(480).
	�� ��������� ������ ��������� ���������� � �������� ������� ���������� �������.
	�������� ���� ��������� ��������� ��� resize ����� ������� �� ����������� ����������.
	�� ��������� � ������ ����������.
	*/
	resize(src, src, Size(640, src.rows / (src.cols*1.) * 640));	
	normalize(src, src, 0, 255, NORM_MINMAX);		// ���������� ����������. ����� ����� ������ ���� ������ �� ����� ���� ����������� ���������

	bilateralFilter(src, blured, kernel_size, kernel_size * 2, kernel_size / 2);	// ������ ������������ ������, �� �������� background
	threshold(blured, src, thresh, 255, THRESH_BINARY);				// ��������� ����������
	morphologyEx(src, src, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(2, 2)));		// ��������� �� ����� ��� ������� ��������

	return src;
}

/*
�������, ��� ��������� ������� ����.
���������� �� 2 ����.
*/
vector<vector<Point>> cartContours(Mat& src) {
	vector<vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);		// ��������� �� �������, ����� � ��������.
	vector<vector<Point>> good_contours;
	double minArea = 0.;

	for (int i = 0; i < contours.size(); i++) {			// ��������� �� ��� ��������� �������� � ������� ���, �� �� ������� �����.
		vector<Point> c = contours.at(i);
		if (c.size() > 30) {							// ����� 1: ������ �� ���� ������ �� 30 �����
			vector<Point> hull;
			convexHull(c, hull);
			auto solidity = static_cast<double>(contourArea(c)) / contourArea(hull);
			if (solidity > solidity_thresh) {			// ����� 2: ������ �� �� ���� ��������.
				approxPolyDP(c, c, arcLength(c, true) / 15, true);
				if (c.size() == 4) {					// ����� 3: ���� ������������ ������� ��������, �� �������� 4 ����� (����)
					int badMark = 0;
					for (int j = 0; j < c.size(); j++) {
						Point a = c.at(j);
						if (a.x < 5 || a.x > mask.cols - 5 ||		// ����� 4: ������ �� �� ����������� � ����� ���������� ������� 5 ������.
							a.y < 5 || a.y > mask.rows - 5) badMark = 1;
					}
					if (!badMark) {
						good_contours.push_back(c);
						minArea += contourArea(c) / arcLength(c, true);
					}
				}	
			}
		}
	}
	minArea /= good_contours.size();
	for (int i = 0; i < good_contours.size(); i++) {				// ����� 5: ��������� ����� ������� �� ���� ��������� ����� ����������� �� ����������.
		double cArea = (contourArea(good_contours.at(i))/ arcLength(good_contours.at(i), true)) / minArea;
		if (cArea < 1./area_thresh || cArea > area_thresh) good_contours.erase(good_contours.begin() + i);
	}

	return good_contours;		// �� ����� ���������� �� ������� �������.
}

/*
�������, ��� ��������� ������� �� ����������� ����������.
���������� �� 2 ����.
*/
vector<vector<Point>> resizeContours(vector<vector<Point>> c) {
	double ratio = 1. * src0.cols / mask.cols;
	vector<vector<Point>> c_resized;
	for (int i = 0; i < c.size(); i++) {
		vector<Point> vec;
		for (int j = 0; j < c.at(i).size(); j++) {
			Point p;
			p.x = c.at(i).at(j).x * ratio;
			p.y = c.at(i).at(j).y * ratio;
			vec.push_back(p);
		}
		c_resized.push_back(vec);
	}
	return c_resized;
}

/*
�������, ��� ������ ������������ ������������ ����� � ���� ����� ���������� �����.
���������� �� 3 ����.
*/
Mat cropPaper(const Mat& image, const vector<Point>& contour) {

	const auto a = norm(contour[0] - contour[1]);
	const auto b = norm(contour[1] - contour[2]);

	const static auto vectorContourToMat = [](const vector<Point>& contour) {
		Mat contour_in_mat(contour);
		contour_in_mat.convertTo(contour_in_mat, CV_32F);
		return contour_in_mat;
	};

	Mat paper(Size(min(a,b),max(a,b)), CV_8UC1);
	Mat tmp1 = vectorContourToMat(contour);
	Mat tmp2;
	/*
	A  B

	D  C
	*/
	if (b > a)		// ���� ������� 0-1 ����� �� 1-2, �� ������ ����������� � ����� D ����� ����������� ������.
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << 0, b, a, b, a, 0, 0, 0).reshape(2);
	if (a > b)		// ���� ������� 0-1 ����� �� 1-2, �� ������ ����������� � ����� � ����� ����������� ������.
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << b, a, b, 0, 0, 0, 0, a).reshape(2);
	auto p_trans_mat = getPerspectiveTransform(tmp1, tmp2);		// ��������� ������� ������������
	warpPerspective(image, paper, p_trans_mat, paper.size());
	return paper;		// ������� ����� ���������� �����
}

int main() {
	src0 = imread(image + "114.jpg", IMREAD_GRAYSCALE);
	if (!src0.data) {
		cout << "File not found";
		return(1);
	}
	/*
	���� 1. �����������.
	input: ����������� ����������
	output: ������ ����� � ����������� ��������� ���� � �������� ����.
	*/
	mask = PreProc(src0);

	/*
	���� 2. �������� ������� ����.
	input: ����� (output ����� 1)
	output: ������ ����, �� ���������� � 4 ����� (����). 
			resize ��� ����� �� ������ ����������� ����������.
	*/
	vector<vector<Point>> contours = cartContours(mask);
	vector<vector<Point>> contoursResized = resizeContours(contours);

	/*
	���� 3. ������������ ������������ �������.
	input: ������� ���� (output ����� 2)
	output: vector<Mat>, �� ����� Mat - ����� ���������� �����.
	*/
	vector<Mat> carts;
	for (int i = 0; i < contoursResized.size(); i++) {
		Mat c = cropPaper(src0, contoursResized.at(i));
		carts.push_back(c);
	}

	/*
	���� 4. (� ������) ��������� ���� � ���������.
	input: ������ ��������� ���� (output ����� 3)
	output: ������� ���������� ������� �������� (�����, ��������)
	*/

	/*
	���� 5. (� ������) �������� ���������
	input: ������ ������� ���� (output ����� 4)
	output: ����� ���������.
	*/

	waitKey(0);
	return(0);
}