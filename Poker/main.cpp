#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const string imgPath = "./Images/";


/*
	The function findBiggestRegion looks for the largest white region of the src and copies it to dst.
*/

void findLargestRegion(Mat &src, Mat &dst)
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

	if (componentsInfo.size() > 0)
	{
		sort(componentsInfo.begin(), componentsInfo.end(), [](const pair<int, int> &a, const pair<int, int> &b) {
			return a.second > b.second;
		});
	}
	
	Mat tmp = Mat::zeros(src.size(), CV_8UC1);

	for (int x = 0; x < src.size().width; x++)
	{
		for (int y = 0; y < src.size().height; y++)
		{
			if (components.at<ushort>(y, x) == componentsInfo[0].first)
			{
				tmp.at<uchar>(y, x) = 255;
			}
		}
	}
	dst = tmp.clone();
}

int main()
{
	Mat img = imread(imgPath + "3.jpg");
	resize(img, img, cv::Size(0, 0), 0.2, 0.2);
	
	Mat imgBlurred;
	GaussianBlur(img, imgBlurred, Size(3, 3), 1);
	
	Mat imgGray;
	cvtColor(imgBlurred, imgGray, CV_BGR2GRAY);
	
	Mat imgBin;
	threshold(imgGray, imgBin, 125, 255, THRESH_BINARY | THRESH_OTSU);

	Mat clearedImg;
	int kernelSize = 1;
	auto kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(2 * kernelSize + 1, 2 * kernelSize + 1));
	morphologyEx(imgBin, clearedImg, CV_MOP_CLOSE, kernel);
	
	findLargestRegion(clearedImg, clearedImg);
	// paint the largest region of clearedImg(card) in white
	for (int x = 0; x < clearedImg.size().width; x++)
	{
		pair<int, int> whiteSegment = { -1,-1 };
		for (int y = 0; y < clearedImg.size().height; y++)
		{
			if (*clearedImg.ptr<uchar>(y, x) > 0)
			{
				if (whiteSegment.first == -1)
				{
					whiteSegment = { y, y };
				}
				else
				{
					whiteSegment.second = y;
				}
			}
		}
		
		for (int y = whiteSegment.first + 1; y < whiteSegment.second; y++)
		{
			clearedImg.at<uchar>(y, x) = 255;
		}
	}

	Mat cardMask = clearedImg.clone();
	Mat result = Mat::zeros(img.size(), CV_8UC1);
	img.copyTo(result, cardMask);
	hconcat(vector<Mat>{img, result}, result);
	imshow("Image", result);

	waitKey();
}