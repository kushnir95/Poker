#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

string image = "d:\\LECTIONS\\_LectionExtract_\\Programming\\CV&AI_Abto\\PROJECT\\Images\\1.jpg";
string mask_image = "d:\\LECTIONS\\_LectionExtract_\\Programming\\CV&AI_Abto\\PROJECT\\Images\\Prepare_Image_masks\\mask_1.jpg";
string writeDir = "d:\\LECTIONS\\_LectionExtract_\\Programming\\CV&AI_Abto\\PROJECT\\Images\\Search_Cart_templates\\";

void imfill(const Mat& src, Mat& dst) {
	Mat src_adjunct = src.clone();
	for (auto seedPoint : vector<Point>{ Point(0, 0),
		Point(dst.cols - 1, 0),
		Point(0, dst.rows - 1),
		Point(dst.cols - 1, dst.rows - 1) }) {
		floodFill(src_adjunct, seedPoint, Scalar(255));
	}
	dst = src | ~src_adjunct;
}

Mat findPaperMask(const Mat& image) {
	Mat mask;
	cvtColor(image, mask, cv::COLOR_BGR2YCrCb);
	imfill(image, mask);
	return mask;
}

vector<Point> findPaperContour(Mat& mask) {
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_L1);
	auto contour = *max_element(contours.begin(), contours.end(), [](const vector<Point>& cnt1, const vector<Point>& cnt2) {
		return contourArea(cnt1) < contourArea(cnt2);
	});
	approxPolyDP(contour, contour, arcLength(contour, true) / 50, true);

	/*vector<Point2f> obj = {Point2f(0,0), Point2f(1,0), Point2f(1,1), Point2f(0,1)};
	vector<Point2f> scene = contour.data;
	Mat H = findHomography(obj, scene, RANSAC);*/

	vector<vector<Point>> cart_contours;
	vector<Point2f> corners;
	float alfa = 3. / 4;
	float xdiv = contour[0].x - contour[(int)contour.size()].x;
	float ydiv = contour[0].y - contour[(int)contour.size()].y;
	float dist = sqrt(xdiv*xdiv + ydiv*ydiv);
	corners.push_back(dist, ydiv/xdiv);
	float max_dist = 0;
	Point corner = contour[(int)contour.size()];
	size_t k;
	for (size_t i = 0; i < contour.size()-1; i++) {
		xdiv = contour[i].x - contour[i+1].x;
		ydiv = contour[i].y - contour[i+1].y;
		dist = sqrt(xdiv*xdiv + ydiv*ydiv);
		corners.push_back(dist, ydiv / xdiv);
		if (max_dist < dist) {
			max_dist = dist;
			corner = contour[i];
			k = i;
		}
	}
	for (size_t i = 0; i < contour.size(); i++) {
		
	}
	cart_contours.push_back({contour[k], contour[k+1], 
							Point(contour[k+1]+alfa*corners[k+1].x/sqrt(1+ corners[k + 1].y)),
							Point(contour[k - 1] + alfa*corners[k - 1].x / sqrt(1 + corners[k - 1].y)) });

	return contour;
}

Mat cropPaper(const Mat& image, const vector<Point>& contour) {
	const static auto resolvePaperSize = [](const vector<Point>& contour) {
		const auto width = norm(contour[0] - contour[1]);
		const auto height = norm(contour[1] - contour[2]);
		return Size(max(width, height), min(width, height));
	};
	const static auto vectorContourToMat = [](const vector<Point>& contour) {
		Mat contour_in_mat(contour);
		contour_in_mat.convertTo(contour_in_mat, CV_32F);
		return contour_in_mat;
	};
	const static auto buildActualPaperContour = [](const Size& size) {
		return static_cast<Mat>(Mat_<float>(8, 1, CV_32F) << 0, 0, size.width, 0, size.width, size.height, 0, size.height).reshape(2);
	};

	Mat paper(image.size(), CV_8UC3);
	Mat tmp1 = vectorContourToMat(contour);
	Mat tmp2 = buildActualPaperContour(paper.size());
	auto p_trans_mat = getPerspectiveTransform(tmp1, tmp2);
	warpPerspective(image, paper, p_trans_mat, paper.size());
	return paper;
}

void main() {
	Mat img = imread(image);
	Mat src = imread(mask_image);
	Mat mask, mask_new, dst;
	mask = findPaperMask(src);
	cvtColor(mask, mask, CV_BGR2GRAY);
	morphologyEx(mask, mask_new, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(15, 15)));
	//blur(mask, mask_new, Size(9, 9));
	threshold(mask_new, mask_new, 240, 255, THRESH_BINARY);
	hconcat(vector<Mat>{mask, mask_new}, dst);
	imshow("win", dst);

	auto contour = findPaperContour(mask_new);

	////if (contour.size() == 4) {
		auto paper_image = cropPaper(img, contour);
		imshow("Paper", paper_image);

		/*int width = paper_image.cols;
		int height = paper_image.rows;
		Point2f center(width / 2.0, height / 2.0);
		Rect bbox = RotatedRect(center, paper_image.size(), -90).boundingRect();
		Mat rot = getRotationMatrix2D(center, -90, 1);
		rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

		Mat compare;
		warpAffine(paper_image, compare, rot, bbox.size(), 1, BORDER_REFLECT);
		flip(compare, compare, 1);
		imshow("Unit", compare);
		imwrite(writeDir + "templ_" + image.substr(image.find_last_of('\\') + 1), compare);*/
	//}

	waitKey(0);
}