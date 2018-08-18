#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string img_1_path = "..\\..\\Images\\Search_Cart_templates\\templ_02.jpg";
string img_2_path = "..\\..\\Images\\Search_Cart_templates\\templ_12.jpg";
string templ_path = "..\\..\\Images\\Search_Cart_templates\\templ_12.jpg";
int max_coefficient = 20;
int epsilon = 9;
Mat templ;

Mat matching(Mat& image) {
	int minHessian = 500;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute(image, Mat(), keypoints_1, descriptors_1);
	detector->detectAndCompute(templ, Mat(), keypoints_2, descriptors_2);
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
		if (matches[i].distance <= max((max_coefficient * min_dist) / 10, 0.02))
		{
			Point2f p1 = keypoints_1[matches[i].queryIdx].pt;
			Point2f p2 = keypoints_2[matches[i].trainIdx].pt;
			float xdiv = p1.x - p2.x;
			float ydiv = p1.y - p2.y;
			float div = sqrt(xdiv*xdiv + ydiv*ydiv);
			if (div < image.cols / (epsilon + 1) && div < image.rows / (epsilon + 1)) {
				good_matches.push_back(matches[i]);
			}
		}
	}
	
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(image, keypoints_1, templ, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return img_matches;
}

long simpleMatching(Mat& image) {
	Mat result;
	resize(image, image, templ.size());
	matchTemplate(image, templ, result, TM_SQDIFF);
	int sum = 0;
	for (int i = 0; i < result.cols; i++) {
		sum += result.at<uchar>(0,i);
	}
	return sum;
}

int main() {
	Mat img_1 = imread(img_1_path, IMREAD_GRAYSCALE);
	Mat img_2 = imread(img_2_path, IMREAD_GRAYSCALE);
	templ = imread(templ_path, IMREAD_GRAYSCALE);

	/*img_1 = img_1(Rect(0, 0, img_1.cols / 3, img_1.rows / 4));
	img_2 = img_2(Rect(0, 0, img_2.cols / 3, img_2.rows / 4));*/
	
	equalizeHist(img_1, img_1);
	equalizeHist(img_2, img_2);
	equalizeHist(templ, templ);

	img_1 = img_1(Rect(0, 0, img_1.cols / 3, img_1.rows / 4));
	img_2 = img_2(Rect(0, 0, img_2.cols / 3, img_2.rows / 4));
	templ = templ(Rect(0, 0, templ.cols / 3, templ.rows / 4));
	if (!img_1.data || !img_2.data || !templ.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	namedWindow("Options", WINDOW_NORMAL);
	namedWindow("Image1", WINDOW_NORMAL);
	namedWindow("Image2", WINDOW_NORMAL);
	createTrackbar("Max Coefficient", "Options", &max_coefficient, 100);
	createTrackbar("Epsilon", "Options", &epsilon, 100);

	//cout << simpleMatching(img_1) << endl;
	//cout << simpleMatching(img_2) << endl;
	while (waitKey(30) != 27) {
		imshow("Image1",matching(img_1));
		imshow("Image2", matching(img_2));
	}
	return 0;
}