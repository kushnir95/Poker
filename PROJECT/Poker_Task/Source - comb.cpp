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

enum Suits { SPADE = 0, CLUB = 1, HEART = 2, DIAMOND = 3 };

vector<Mat> vec_patterns;
vector<vector<Point>> pattern_contours;
map<int, vector<double>> pattern_descriptors;

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
Параметри, що використовуються у функції PreProc().
blurK - розмір ядра білатерального фільтра.
thresh - границя кольору при бінаризації.
*/
int blurK = 8;
int thresh = 150;

/*
Параметри, що використовуються у функції cartContours().
solidity_thresh = 0.9 - означає, що виживають лише контури, у яких solidity > 0.9 (не увігнуті контури).
						solidity - це відношення площі контура до площі convexHull.
area_thresh - максимальне відхилення rec від середнього значення по контурам.
				rec - це відношення площі контура до його периметра.
*/
double solidity_thresh = 0.9;
double area_thresh = 1.5;

int max_coefficient = 20;
int epsilon = 1;

/*
Функція яка препроцесить зображення.
Застосовна на 1 етапі.
*/
Mat PreProc(Mat src) {
	Mat blured;
	int kernel_size = blurK * 2 + 1;		// робимо розмір ядра непарним числом

	/*
	Зменшуємо зображення до 640х(480).
	Це дозволить швидко процесити зображення і зменшити кількість непотрібних контурів.
	Натомість може виникнути неточність при resize точок контура до нормального зображення.
	Цю неточність я вважаю прийнятною.
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
Функція, яка знаходить контури карт.
Застосовна на 2 етапі.
*/
vector<vector<Point>> cartContours(Mat& src) {
	vector<vector<Point>> contours;
	findContours(src, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);		// Знаходимо всі контури, перші в ієрархії.
	vector<vector<Point>> good_contours;
	double minArea = 0.;

	for (int i = 0; i < contours.size(); i++) {			// Проходимо по всім знайдених контурам і відсіюємо тих, що не пройшли умови.
		vector<Point> c = contours.at(i);
		if (c.size() > 10) {							// Умова 1: контур має бути більшим від 30 точок
			vector<Point> hull;
			convexHull(c, hull);
			auto solidity = static_cast<double>(contourArea(c)) / contourArea(hull);
			if (solidity > solidity_thresh) {			// Умова 2: контур не має бути увігнутим.
				approxPolyDP(c, c, arcLength(c, true) / 15, true);
				if (c.size() == 4) {					// Умова 3: після апроксимації прямими відрізками, має лишитись 4 точки (кути)
					int badMark = 0;
					for (int j = 0; j < c.size(); j++) {
						Point a = c.at(j);
						if (a.x < 5 || a.x > mask.cols - 5 ||		// Умова 4: контур не має знаходитись в рамці зображення розміром 5 пікселів.
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
	for (int i = 0; i < good_contours.size(); i++) {				// Умова 5: відношення площі контуру до його периметра слабо відрізняється від середнього.
		double cArea = (contourArea(good_contours.at(i)) / arcLength(good_contours.at(i), true)) / minArea;
		if (cArea < 1. / area_thresh || cArea > area_thresh) good_contours.erase(good_contours.begin() + i);
	}

	return good_contours;		// На вихід поступають всі виживші контури.
}

/*
Функція, яка ресайзить контури до початкового зображення.
Застосовна на 2 етапі.
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
Функція, яка виконує перспективне перетворення карти і шукає пряме зображення карти.
Застосовна на 3 етапі.
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
	if (b > a)		// Якщо сторона 0-1 менша від 1-2, то будуємо прямокутник з точки D проти годинникової стрілки.
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << 0, b, a, b, a, 0, 0, 0).reshape(2);
	if (a > b)		// Якщо сторона 0-1 більша від 1-2, то будуємо прямокутник з точки С проти годинникової стрілки.
		tmp2 = static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << b, a, b, 0, 0, 0, 0, a).reshape(2);
	auto p_trans_mat = getPerspectiveTransform(tmp1, tmp2);		// Знаходимо матрицю перетворення
	warpPerspective(image, paper, p_trans_mat, paper.size());
	return paper;		// Вертаємо пряме зображення карти
}

vector<Point> prepare_contour(Mat& image) {
	Mat th = Mat::zeros(image.size(), CV_8UC1);
	threshold(image, th, 170, 255, THRESH_BINARY_INV);
	morphologyEx(th, th, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(2, 2)));
	vector<vector<Point>> contours;
	findContours(th, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int cArea, maxArea = 0, maxArea_idx = 0;
	if (contours.size() != 1)
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> c = contours.at(j);
			int badPoint = 0;
			for (int i = 0; i < c.size(); i++) {
				Point p = c.at(i);
				if (p.x < 5 || p.x > image.cols - 5 || p.y < 5 || p.y > image.rows - 5) {
					badPoint = 1;
					break;
				}
			if(badPoint) contours.erase(contours.begin() + j);
			else {
			cArea = contourArea(c);
			if (cArea > maxArea) {
				maxArea = cArea;
				maxArea_idx = j;
				}
			}
			}
		}
	return contours.at(maxArea_idx);
}

Vec2d calculateCoef(vector<Point>& c) {
	vector<Point> hull;
	convexHull(c, hull);
	return { arcLength(hull, true) / arcLength(c, true) , contourArea(c) / arcLength(c, true) };
}

vector<vector<Point>> split_contour(vector<Point>& c) {
	int minY = 500;
	int maxY = 0;
	int minX = 500;
	int maxX = 0;

	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.y < minY) minY = p.y;
		if (p.y > maxY) maxY = p.y;

		if (p.x < minX) minX = p.x;
		if (p.x > maxX) maxX = p.x;
	}
	int meanY = (maxY + minY) / 2;
	int meanX = (maxX + minX) / 2;
	vector<Point> first, second, third, fourth;
	for (int i = 0; i < c.size(); i++) {
		Point p = c.at(i);
		if (p.x >= meanX && p.y < meanY) first.push_back(p);
		if (p.x < meanX && p.y < meanY) second.push_back(p);
		if (p.x < meanX && p.y >= meanY) third.push_back(p);
		if (p.x >= meanX && p.y >= meanY) fourth.push_back(p);
	}
	return { first, second, third, fourth };
}

void prepare_pattern_contours() {
	for (Mat pat : vec_patterns) {
		pattern_contours.push_back(prepare_contour(pat));
	}
}

void prepare_pattern_descriptors() {
	for (int i = 0; i < pattern_contours.size(); i++) {
		vector<Point> pattern_contour = pattern_contours.at(i);
		vector<vector<Point>> splited = split_contour(pattern_contour);
		vector<double> descriptors = { calculateCoef(pattern_contour)[0], calculateCoef(pattern_contour)[1], 
										calculateCoef(splited.at(0))[0], calculateCoef(splited.at(0))[1], 
										calculateCoef(splited.at(1))[0], calculateCoef(splited.at(1))[1],
										calculateCoef(splited.at(2))[0], calculateCoef(splited.at(2))[1],
										calculateCoef(splited.at(3))[0], calculateCoef(splited.at(3))[1], };
		pattern_descriptors.insert(pair<int, vector<double>>(i, descriptors));
	}
}

double match(vector<double>& img_desc, int key) {

	vector<double> descriptors = pattern_descriptors.find(key)->second;

	double ratio1 = abs(img_desc[0] / descriptors[0] - 1);
	double ratio2 = abs(img_desc[1] / descriptors[1] - 1);
	double first_ratio1 = abs(img_desc[2] / descriptors[2] - 1);
	double first_ratio2 = abs(img_desc[3] / descriptors[3] - 1);
	double second_ratio1 = abs(img_desc[4] / descriptors[4] - 1);
	double second_ratio2 = abs(img_desc[5] / descriptors[5] - 1);
	double third_ratio1 = abs(img_desc[6] / descriptors[6] - 1);
	double third_ratio2 = abs(img_desc[7] / descriptors[7] - 1);
	double fourth_ratio1 = abs(img_desc[8] / descriptors[8] - 1);
	double fourth_ratio2 = abs(img_desc[9] / descriptors[9] - 1);

	return ratio1 + ratio2 + first_ratio1 + first_ratio2 + second_ratio1 
		+ second_ratio2 + third_ratio1 + third_ratio2 + fourth_ratio1 + fourth_ratio2;
}

Card matchCart(Mat cart) {
	Mat part = cart(Rect(0, 10, cart.cols / 5, cart.rows / 3));
	Mat part_grey;
	cvtColor(part, part_grey, COLOR_BGR2GRAY);

	int cart_value = 2, cart_suit = 0;
	Mat part_value = part_grey(Rect(0, 0, part.cols, part.rows / 2));
	Mat part_suit = part_grey(Rect(0, part.rows / 2, part.cols, part.rows / 2));

	vector<Point> image_contour = prepare_contour(part_suit);
	double img_coef1 = calculateCoef(image_contour)[0];
	double img_coef2 = calculateCoef(image_contour)[1];

	vector<vector<Point>> splited = split_contour(image_contour);
	double first_coef1 = calculateCoef(splited.at(0))[0];
	double first_coef2 = calculateCoef(splited.at(0))[1];

	double second_coef1 = calculateCoef(splited.at(1))[0];
	double second_coef2 = calculateCoef(splited.at(1))[1];

	double third_coef1 = calculateCoef(splited.at(2))[0];
	double third_coef2 = calculateCoef(splited.at(2))[1];

	double fourth_coef1 = calculateCoef(splited.at(3))[0];
	double fourth_coef2 = calculateCoef(splited.at(3))[1];

	vector<double> desc_image = { img_coef1, img_coef1, first_coef1, first_coef2, second_coef1, second_coef2,
									third_coef1, third_coef2, fourth_coef1, fourth_coef2};

	double minNum = DBL_MAX;
	int idx_of_minNum = 0;
	for (int i = 0; i < 4; i++) {
		double value = match(desc_image, i);
		if (value < minNum) {
			minNum = value;
			idx_of_minNum = i;
		}
	}
	cart_suit = idx_of_minNum;

	image_contour = prepare_contour(part_suit);
	img_coef1 = calculateCoef(image_contour)[0];
	img_coef2 = calculateCoef(image_contour)[1];

	splited = split_contour(image_contour);
	first_coef1 = calculateCoef(splited.at(0))[0];
	first_coef2 = calculateCoef(splited.at(0))[1];

	second_coef1 = calculateCoef(splited.at(1))[0];
	second_coef2 = calculateCoef(splited.at(1))[1];

	third_coef1 = calculateCoef(splited.at(2))[0];
	third_coef2 = calculateCoef(splited.at(2))[1];

	fourth_coef1 = calculateCoef(splited.at(3))[0];
	fourth_coef2 = calculateCoef(splited.at(3))[1];

	desc_image = { img_coef1, img_coef1, first_coef1, first_coef2, second_coef1, second_coef2,
									third_coef1, third_coef2, fourth_coef1, fourth_coef2 };

	minNum = DBL_MAX;
	idx_of_minNum = 4;
	for (int i = 4; i < pattern_contours.size(); i++) {
		double value = match(desc_image, i);
		if (value < minNum) {
			minNum = value;
			idx_of_minNum = i;
		}
	}
	cart_value = idx_of_minNum - 2;

	return Card(cart_suit, cart_value);
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
	src0 = imread(image + "101.jpg");
	if (!src0.data) {
		cout << "File not found";
		return(1);
	}
	/*
	Етап 1. Препроцесінг.
	input: завантажене зображення
	output: бінарна маска з окресленими силуетами карт і можливих бліків.
	*/
	mask = PreProc(src0);

	/*
	Етап 2. Виділення контурів карт.
	input: маска (output етапу 1)
	output: контур карт, що складається з 4 точок (кутів).
			resize цих точок до розміру початкового зображення.
	*/
	vector<vector<Point>> contours = cartContours(mask);
	vector<vector<Point>> contoursResized = resizeContours(contours);

	/*
	Етап 3. Перспективне перетворення контурів.
	input: контури карт (output етапу 2)
	output: vector<Mat>, де кожен Mat - пряме зображення карти.
	*/
	vector<Mat> carts;
	for (int i = 0; i < contoursResized.size(); i++) {
		Mat c = cropPaper(src0, contoursResized.at(i));
		carts.push_back(c);
	}

	/*
	Етап 4. (В процесі) Порівняння карт з шаблонами.
	input: вектор зображень карт (output етапу 3)
	output: кожному зображенню присвоєні значення (масть, значення)
	*/
	upload_Patterns();
	prepare_pattern_contours();
	prepare_pattern_descriptors();
	vector<Card> current_comb;
	for (int i = 0; i < carts.size(); i++) {
		Card cart_info = matchCart(carts.at(i));
		cout << cart_info.toString() << endl;
		current_comb.push_back(cart_info);
	}

	/*
	Етап 5. Перевірка комбінацій
	input: вектор значень карт (output етапу 4)
	output: назва комбінації.
	*/
	Card c;
	string combo = c.getCombination(current_comb);
	cout << combo << endl;

	/*
	Етап 6. Виведення на екран зображення з контурами і написами
	*/
	//buildUI(contours, current_comb);


	waitKey(0);
	return(0);
}