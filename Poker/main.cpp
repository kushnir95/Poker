#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const string imgPath = "./Images/five/";
const int cardWidth = 200;
const int cardHeight = 300;

/*
The function findBiggestRegion looks for the largest white region of the src and copies it to dst.
*/

void findRegions(Mat &src, Mat &dst, int amountOfRegions)
{
	Mat components = Mat::zeros(src.size(), CV_16UC1);
	int componentIdx = 0;
	vector < pair<int, int> > componentsInfo;
	for (int x = 0; x < src.size().width; x++)
	{
		for (int y = 0; y < src.size().height; y++)
		{
			if ((*components.ptr<ushort>(y, x) == 0) && (*src.ptr<uchar>(y, x) > 0))
			{
				queue< pair<int, int> > q;
				q.push({ x, y });
				componentIdx++;
				components.at<ushort>(y, x) = componentIdx;
				int componentSize = 1;
				while (!q.empty())
				{
					pair<int, int> currentPixel = q.front();

					q.pop();
					const int dx[] = { 1, 0, 0 ,-1 };
					const int dy[] = { 0, 1, -1, 0 };
					for (int dir = 0; dir < 4; dir++)
					{
						pair<int, int> neighbourPixel = { currentPixel.first + dx[dir], currentPixel.second + dy[dir] };
						if ((neighbourPixel.first >= 0) && (neighbourPixel.first < src.size().width)
							&& (neighbourPixel.second >= 0) && (neighbourPixel.second < src.size().height)
							&& (components.at<ushort>(neighbourPixel.second, neighbourPixel.first) == 0)
							&& (*src.ptr<uchar>(neighbourPixel.second, neighbourPixel.first) > 0))
						{
							componentSize++;
							components.at<ushort>(neighbourPixel.second, neighbourPixel.first) = componentIdx;
							q.push({ neighbourPixel.first, neighbourPixel.second });
						}
					}
				}
				componentsInfo.push_back({ componentIdx, componentSize });
			}
		}
	}

	Mat tmp = src.clone();

	if (componentsInfo.size() > 0)
	{
		sort(componentsInfo.begin(), componentsInfo.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
			return a.second > b.second;
		});
	}
	else
	{
		dst = tmp.clone();
		return;
	}


	set<int> interestRegions;
	for (int i = 0; i < min((int)componentsInfo.size(), amountOfRegions); i++)
	{
		interestRegions.insert(componentsInfo[i].first);
	}

	for (int x = 0; x < src.size().width; x++)
	{
		for (int y = 0; y < src.size().height; y++)
		{
			if (interestRegions.find(components.at<ushort>(y, x)) == interestRegions.end())
			{
				tmp.at<uchar>(y, x) = 0;
			}
		}
	}
	dst = tmp.clone();
}

void findCardsContours(const Mat& image, vector<vector<Point> > &contours, vector<Vec4i> &hierarchy) {
	Mat mask;
	cvtColor(image, mask, cv::COLOR_BGR2YCrCb);
	vector<Mat> mask_planes;
	split(mask, mask_planes);
	Mat binarizedImg;
	threshold(mask_planes[0], binarizedImg, 175, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	findRegions(binarizedImg, binarizedImg, 5);
	findContours(binarizedImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}

int getDistance(const Point& firstPoint, const Point& secondPoint)
{
	return (firstPoint.x - secondPoint.x) * (firstPoint.x - secondPoint.x) + (firstPoint.y - secondPoint.y) * (firstPoint.y - secondPoint.y);
}

int main()
{
	Mat img = imread(imgPath + "1.jpg");
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findCardsContours(img, contours, hierarchy);

	//Drawing contours
	Mat result = img.clone();
	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		drawContours(result, contours, idx, Scalar(0, 255, 0), 5, 8, hierarchy);
	}

	//Approximate contours
	vector<vector<Point> > approximatedContours;
	vector<Mat> cards;
	Mat cImg = img.clone();
	vector<Scalar> color = { Scalar(255, 0, 0), Scalar(0, 255, 0) , Scalar(0, 0, 255) , Scalar(0, 0, 0) };
	for (int idx = 0; idx < contours.size(); idx++)
	{
		vector<Point> approxVector;
		approxPolyDP(contours[idx], approxVector, 15.0, true);
		approximatedContours.push_back(approxVector);
		
		if (approxVector.size() == 4)
		{
			int dist1 = getDistance(approxVector[0], approxVector[1]);
			int dist2 = getDistance(approxVector[1], approxVector[2]);
			Point2f dstPTPoints[4], srcPTPoints[4];
			
			for (int i = 0; i < 4; i++)
			{
				circle(cImg, approxVector[i], 5, color[i], 2);
				srcPTPoints[i] = approxVector[i];
			}

			dstPTPoints[0] = Point2f(0, 0);
			dstPTPoints[2] = Point2f(cardWidth, cardHeight);


			if (dist1 < dist2)
			{
				dstPTPoints[1] = Point2f(cardWidth, 0);
				dstPTPoints[3] = Point2f(0, cardHeight);
			}
			else
			{
				dstPTPoints[1] = Point2f(0, cardHeight);
				dstPTPoints[3] = Point2f(cardWidth, 0);
			}

			Mat PTMatrix = getPerspectiveTransform(srcPTPoints, dstPTPoints);
			Mat extractedCardImg = Mat::zeros(Size(cardWidth, cardHeight), CV_8UC3);
			warpPerspective(img, extractedCardImg, PTMatrix, extractedCardImg.size());

			cards.push_back(extractedCardImg);
		}
	}

	result = cards[0].clone();
	for (int idx = 1; idx < cards.size(); idx++)
	{
		hconcat(vector<Mat>{result, cards[idx]}, result);
	}
	cv::imshow("Image with contours", cImg);
	cv::imshow("Origin image", img);
	cv::imshow("Result", result);
	cv::waitKey();
}