#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <map>

#include "Card.h"
#include "Features_Mapper.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string image = "..\\..\\Images\\";
string imgName = "101.jpg";

Ptr<ml::KNearest> SuitKNN = ml::StatModel::load<ml::KNearest>("./KNN_model/SPADE.xml");
Ptr<ml::KNearest> EightKNN = ml::StatModel::load<ml::KNearest>("./KNN_model/EIGHT.xml");
Ptr<ml::KNearest> SixKNN = ml::StatModel::load<ml::KNearest>("./KNN_model/SIX.xml");
Ptr<ml::KNearest> TwoKNN = ml::StatModel::load<ml::KNearest>("./KNN_model/TWO.xml");

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
double area_thresh = 1.5;

int max_coefficient = 20;
int epsilon = 1;

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
	threshold(mask_planes[0], src, 145, 255, CV_THRESH_BINARY);
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

Vec2i contourKids(Mat& image) {
	Mat img_th;
	threshold(image, img_th, 150, 255, THRESH_BINARY_INV);
	morphologyEx(img_th, img_th, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	vector<vector<Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(img_th, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	int maxArea = 0;
	int maxArea_idx = 0;
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
	vector<vector<Point>> kids;
	for (int j = 0; j < hierarchy.size(); j++) {
		if (hierarchy.at(j)[3] == maxArea_idx) kids.push_back(contours.at(j));
	}
	return { maxArea, (int)kids.size() };
}

vector<Mat> generate(Mat& image, int nInstances) {
	vector<Mat> imgs;
	imgs.push_back(image);
	for (int k = 0; k < nInstances; k++) {
		Mat genImg;
		double angleValue = rand() % (5 + 1);
		if (rand() % 2 > 0) angleValue *= -1.0;

		Mat rotationMatrix2D = getRotationMatrix2D(Point(image.cols / 2, image.rows / 2), angleValue, 1);
		warpAffine(image, genImg, rotationMatrix2D, image.size());
		imgs.push_back(genImg);
	}
	return imgs;
}

int findFrequentlyClass(Mat& predict) {
	int maxFreq = 0, maxFreq_idx = 0;
	for (int j = 0; j < predict.rows; j++) {
		int clas = (int)predict.at<float>(j, 0);
		int freq = 0;
		for (int y = j + 1; y < predict.rows; y++) {
			if ((int)predict.at<float>(y, 0) == clas)
				freq++;
		}
		if (freq > maxFreq) {
			maxFreq = freq;
			maxFreq_idx = clas;
		}
	}
	return maxFreq_idx;
}

int match(Mat& image, const Ptr<cv::ml::KNearest>& Two, const Ptr<cv::ml::KNearest>& One, const Ptr<cv::ml::KNearest>& Zero) {
	vector<Mat> ranks = generate(image, 5);
	vector<Mat> rank_desc;
	for (Mat m : ranks) {
		rank_desc.push_back(Features_Mapper::prepare_descriptors(m, 1));
	}

	Mat img_th;
	threshold(image, img_th, 150, 255, THRESH_BINARY_INV);
	morphologyEx(img_th, img_th, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
	vector<vector<Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(img_th, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	int maxArea = 0;
	int maxArea_idx = 0;
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
	vector<vector<Point>> kids;
	for (int j = 0; j < hierarchy.size(); j++) {
		if (hierarchy.at(j)[3] == maxArea_idx) kids.push_back(contours.at(j));
	}
	Mat predict;
	if (kids.size() == 2) {
		Two->findNearest(rank_desc, 100, predict);
	}
	else if (kids.size() == 1) {
		One->findNearest(rank_desc, 80, predict);
	}
	else Zero->findNearest(rank_desc, 50, predict);

	int card_rank = findFrequentlyClass(predict);

	return card_rank;
}

Card matchCart(Mat image, int idx) {
	cvtColor(image, image, COLOR_BGR2GRAY);
	normalize(image, image, 0, 255, NORM_MINMAX);
	Mat part = image(Rect(0, 15, image.cols / 5, image.rows / 3));
	Mat part2 = part.clone();

	int card_rank = 4, card_suit = 0;
	Mat part_rank1 = part(Rect(0, 0, part.cols, part.rows / 2));
	Mat part_suit = part(Rect(0, part.rows / 2, part.cols, part.rows / 2));

	vector<Mat> suits = generate(part_suit, 5);
	vector<Mat> suit_desc;
	for (Mat m : suits) {
		suit_desc.push_back(Features_Mapper::prepare_descriptors(m, 0));
	}
	Mat predict;
	SuitKNN->findNearest(suit_desc, 60, predict);
	card_suit = findFrequentlyClass(predict);
	
	int rank_kids = 0;
	Mat part_rank;
	Mat part_rank2 = part2(Rect(0, part.rows/11, part.cols, part.rows / 2));
	
	Vec2i rank_kids1 = contourKids(part_rank1);
	Vec2i rank_kids2 = contourKids(part_rank2);

	if (rank_kids1[0] <= rank_kids2[0] && rank_kids1[1] < rank_kids2[1]) {
		rank_kids = rank_kids2[1];
		part_rank = part_rank2;
	}
	else {
		rank_kids = rank_kids1[1];
		part_rank = part_rank1;
	}

	vector<Mat> ranks = generate(part_rank, 5);
	vector<Mat> rank_desc;
	for (Mat m : ranks) {
		rank_desc.push_back(Features_Mapper::prepare_descriptors(m, 1));
	}

	if (rank_kids == 2) {
		EightKNN->findNearest(rank_desc, 100, predict);
	}
	else if (rank_kids == 1) {
		SixKNN->findNearest(rank_desc, 80, predict);
	}else TwoKNN->findNearest(rank_desc, 50, predict);

	card_rank = findFrequentlyClass(predict);

	return Card(card_suit, card_rank);
}

void buildUI(vector<vector<Point>> contours, vector<Card> cards) {
	namedWindow("Frame", WINDOW_NORMAL);
	/*for (int i = 0; i < contours.size() - 1; i++) {
		Point a = contours.at(i);
		Point b = contours.at(i + 1);
		cvDrawLine(src0, a, b, Scalar(0, 255, 0));
	}*/
	drawContours(src0, contours, -1, Scalar(0, 255, 0), 4);
	/*for (int i = 0; i < cards.size() - 1; i++) {
		addText(src0, cards.at(i).toString(), contours.at(i).at(0), "", 20);
	}
	Card c;*/
	//addText(src0, c.getCombination(cards), Point(src0.cols/3, 50), "", 30);
	imshow("Frame", src0);
}

int main() {
	src0 = imread("d:/LECTIONS/_LectionExtract_/Programming/CV&AI_Abto/PROJECT/Images/101.jpg");
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

	vector<Card> current_comb;
	for (int i = 0; i < carts.size(); i++) {
		Card cart_info = matchCart(carts.at(i), i);
		cout << cart_info.toString() << endl;
		current_comb.push_back(cart_info);
	}

	/*
	���� 5. �������� ���������
	input: ������ ������� ���� (output ����� 4)
	output: ����� ���������.
	*/

	cout << Card::getCombination(current_comb) << endl;

	/*
	���� 6. ��������� �� ����� ���������� � ��������� � ��������
	*/
	//buildUI(contours, current_comb);


	waitKey(0);
	return(0);
}