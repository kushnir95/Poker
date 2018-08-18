#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

string image = "..\\..\\Images\\";

Mat src0, mask;

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
double area_thresh = 1.3;

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
	normalize(src, src, 0, 255, NORM_MINMAX);		// нормалізуємо зображення. Таким чином робимо більш стійким до різних умов глобального освітлення

	bilateralFilter(src, blured, kernel_size, kernel_size * 2, kernel_size / 2);	// кидаємо білатеральний фільтр, що згладить background
	threshold(blured, src, thresh, 255, THRESH_BINARY);				// бінаризуємо зображення
	morphologyEx(src, src, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(2, 2)));		// прибираємо білі пікселі біля великих областей

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
		if (c.size() > 30) {							// Умова 1: контур має бути більшим від 30 точок
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
		double cArea = (contourArea(good_contours.at(i))/ arcLength(good_contours.at(i), true)) / minArea;
		if (cArea < 1./area_thresh || cArea > area_thresh) good_contours.erase(good_contours.begin() + i);
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

	Mat paper(Size(min(a,b),max(a,b)), CV_8UC1);
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

int main() {
	src0 = imread(image + "114.jpg", IMREAD_GRAYSCALE);
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

	/*
	Етап 5. (В процесі) Перевірка комбінацій
	input: вектор значень карт (output етапу 4)
	output: назва комбінації.
	*/

	waitKey(0);
	return(0);
}