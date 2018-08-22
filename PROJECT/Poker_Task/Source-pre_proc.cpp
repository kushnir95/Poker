#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <map>

#include "Card.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

//string image = "d:\\LECTIONS\\_LectionExtract_\\Programming\\CV&AI_Abto\\Poker-master\\Poker\\Poker\\Images\\five\\";
string image = "..\\..\\Images\\";
string patterns = "..\\..\\Images\\Patterns\\";

Mat src0, mask;

vector<Mat> vec_patterns;

void upload_Patterns() {
	Mat heart = imread(patterns + "heart.jpg", IMREAD_GRAYSCALE);
	Mat diamond = imread(patterns + "diamond.jpg", IMREAD_GRAYSCALE);
	Mat club = imread(patterns + "club.jpg", IMREAD_GRAYSCALE);
	Mat spade = imread(patterns + "spade.jpg", IMREAD_GRAYSCALE);

	Mat two = imread(patterns + "two.jpg", IMREAD_GRAYSCALE);
	Mat three = imread(patterns + "three.jpg", IMREAD_GRAYSCALE);
	Mat four = imread(patterns + "four.jpg", IMREAD_GRAYSCALE);
	Mat five = imread(patterns + "five.jpg", IMREAD_GRAYSCALE);
	Mat six = imread(patterns + "six.jpg", IMREAD_GRAYSCALE);
	Mat seven = imread(patterns + "seven.jpg", IMREAD_GRAYSCALE);
	Mat eight = imread(patterns + "eight.jpg", IMREAD_GRAYSCALE);
	Mat nine = imread(patterns + "nine.jpg", IMREAD_GRAYSCALE);
	Mat ten = imread(patterns + "ten.jpg", IMREAD_GRAYSCALE);
	Mat jack = imread(patterns + "jack.jpg", IMREAD_GRAYSCALE);
	Mat queen = imread(patterns + "queen.jpg", IMREAD_GRAYSCALE);
	Mat king = imread(patterns + "king.jpg", IMREAD_GRAYSCALE);
	Mat ace = imread(patterns + "ace.jpg", IMREAD_GRAYSCALE);

	vec_patterns = { spade, heart, club, diamond, two, three, four, five,
				six, seven, eight, nine, ten, jack, queen, king, ace };
}

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
double area_thresh = 1.5;

int max_coefficient = 5;
int epsilon = 4;

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

	cvtColor(src, src, cv::COLOR_BGR2YCrCb);
	vector<Mat> mask_planes;
	split(src, mask_planes);
	normalize(mask_planes[0], mask_planes[0], 0, 255, NORM_MINMAX);
	threshold(mask_planes[0], src, 140, 255, CV_THRESH_BINARY);
	morphologyEx(src, src, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
	morphologyEx(src, src, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(2, 2)));

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
		if (c.size() > 10) {							// ����� 1: ������ �� ���� ������ �� 30 �����
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
		double cArea = (contourArea(good_contours.at(i)) / arcLength(good_contours.at(i), true)) / minArea;
		if (cArea < 1. / area_thresh || cArea > area_thresh) good_contours.erase(good_contours.begin() + i);
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

	Mat paper(Size(min(a, b), max(a, b)), CV_8UC1);
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

void prepare_patterns() {
	for (Mat m : vec_patterns) {
		threshold(m, m, 150, 255, THRESH_BINARY_INV);
	}
}

int match(Mat& image, Mat& pattern) {
	resize(image, image, pattern.size());
	//threshold(image, image, 150, 255, THRESH_BINARY_INV);

	int minHessian = 500;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute(image, Mat(), keypoints_1, descriptors_1);
	detector->detectAndCompute(pattern, Mat(), keypoints_2, descriptors_2);

	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max((max_coefficient * min_dist), 0.02))
		{
			Point2f p1 = keypoints_1[matches[i].queryIdx].pt;
			Point2f p2 = keypoints_2[matches[i].trainIdx].pt;
			float xdiv = p1.x - p2.x;
			float ydiv = p1.y - p2.y;
			float div = sqrt(xdiv*xdiv + ydiv * ydiv);
			if (div < image.cols / (epsilon + 1) && div < image.rows / (epsilon + 1)) {
				good_matches.push_back(matches[i]);
			}
		}
	}
	Mat outImg;
	drawMatches(image, keypoints_1, pattern, keypoints_2, good_matches, outImg);
	return good_matches.size();
}

Card matchCart(Mat cart) {
	Mat part = cart(Rect(0, 10, cart.cols / 5, cart.rows / 3));
	Mat part_grey;
	cvtColor(part, part_grey, COLOR_BGR2GRAY);

	int cart_value = 2, cart_suit = 0;
	Mat part_value = part_grey(Rect(0, 0, part.cols, part.rows / 2));
	Mat part_suit = part_grey(Rect(0, part.rows / 2, part.cols, part.rows / 2));

	threshold(part_value, part_value, 150, 255, THRESH_BINARY_INV);
	threshold(part_suit, part_suit, 150, 255, THRESH_BINARY_INV);

	double maxNum = 0;
	int idx_of_maxNum = 0;
	for (int i = 0; i < 4; i++) {
		double value = match(part_suit, vec_patterns.at(i));
		if (value > maxNum) {
			maxNum = value;
			idx_of_maxNum = i;
		}
	}
	cart_suit = idx_of_maxNum;

	maxNum = 0;
	idx_of_maxNum = 4;
	for (int i = 4; i < vec_patterns.size(); i++) {
		double value = match(part_value, vec_patterns.at(i));
		if (value > maxNum) {
			maxNum = value;
			idx_of_maxNum = i;
		}
	}
	cart_value = idx_of_maxNum - 2;

	return Card(cart_suit, cart_value);
}

int main() {
	src0 = imread(image + "101.jpg");
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

	upload_Patterns();
	prepare_patterns();
	vector<Card> current_comb;
	for (int i = 0; i < carts.size(); i++) {
		Card cart_info = matchCart(carts.at(i));
		cout << cart_info.toString() << endl;
		current_comb.push_back(cart_info);
	}


	/*
	���� 5. (� ������) �������� ���������
	input: ������ ������� ���� (output ����� 4)
	output: ����� ���������.
	*/

	waitKey(0);
	return(0);
}