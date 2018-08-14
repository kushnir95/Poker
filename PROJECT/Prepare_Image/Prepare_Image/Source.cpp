#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

string image = "..\\..\\Images\\02.jpg";
string output = "..\\..\\Images\\Prepare_Image_masks\\";

Mat src0, src, equalized;

void Customize() {
	
	namedWindow("Image", WINDOW_NORMAL);
	namedWindow("Options", WINDOW_NORMAL);
	int bright = 100;
	int contrast = 100;
	int blurK = 0;
	int sharp = 0;
	int kernel_size;

	int lowest = 20;
	int ratio = 2;

	createTrackbar("Brightness", "Options", &bright, 200);
	createTrackbar("Contrast", "Options", &contrast, 300);
	createTrackbar("Blur", "Options", &blurK, 10);
	createTrackbar("Sharp", "Options", &sharp, 500);
	createTrackbar("Canny_Th", "Options", &lowest, 200);
	createTrackbar("Canny_R", "Options", &ratio, 10);

	Mat blured;
	Mat src_edge, equalized_edge;
	Mat result, result1, result2;
	vector<Mat> lab;

	while (waitKey(30) != 27) {
		src = (src0 + bright - 100) * (contrast / 100.0);

		kernel_size = blurK * 2 + 1;
		bilateralFilter(src, blured, kernel_size, kernel_size*2, kernel_size / 2);
		src = src * (sharp / 100.0) + blured * ((100 - sharp) / 100.0);

		equalizeHist(src, equalized);

		Canny(src, src_edge, lowest, lowest*ratio);
		Canny(equalized, equalized_edge, lowest, lowest*ratio);

		hconcat(vector<Mat>{src, equalized}, result1);
		hconcat(vector<Mat>{src_edge, equalized_edge}, result2);
		vconcat(vector<Mat>{result1, result2}, result);
		imshow("Image", result);
	}
	destroyWindow("Options");
}

void ThresholdingAndMorph() {
	namedWindow("Options", WINDOW_NORMAL);
	namedWindow("Images", WINDOW_NORMAL);

	int thresh = 0;
	createTrackbar("threshold", "Options", &thresh, 255);

	int kernel = 0;
	createTrackbar("kernel_size", "Options", &kernel, 10);

	Mat src_th, equalized_th;
	Mat close_src, close_equ;
	Mat dst, dst1, dst2;

	while (waitKey(30) != 27) {
		threshold(src, src_th, thresh, 255, THRESH_BINARY);
		threshold(equalized, equalized_th, thresh, 255, THRESH_BINARY);

		Mat element = getStructuringElement(MORPH_RECT, Size(2 * kernel + 1, 2 * kernel + 1));
		morphologyEx(src_th, close_src, MORPH_CLOSE, element);
		morphologyEx(equalized_th, close_equ, MORPH_CLOSE, element);

		hconcat(vector<Mat>{src_th, equalized_th}, dst1);
		hconcat(vector<Mat>{close_src, close_equ}, dst2);
		vconcat(vector<Mat>{dst1, dst2}, dst);
		imshow("Images", dst);
	}
	imwrite(output + "mask_" + image.substr(image.find_last_of('\\') + 1) , close_src);
	destroyWindow("Options");
}

void main() {
	src0 = imread(image, IMREAD_GRAYSCALE);
	Customize();
	ThresholdingAndMorph();
	waitKey(0);
}