#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "Features_Mapper.h"

using namespace std;
using namespace cv;

const string imgPath = "./Images/five/";
const string trainsetSuitsDir = "./trainset/suits";
const string trainsetRanksDir = "./trainset/ranks";
const int cardWidth = 200;
const int cardHeight = 300;
const int origialSuitWidth = 40;
const int originalSuitHeight = 40;
const int scaledSuitWidth = 20;
const int scaledSuitHeight = 20;

const int originalRankWidth = 35;
const int originalRankHeight = 55;
const int scaledRankWidth = 20;
const int scaledRankHeight = 40;
const int nRank = 4;

const int nclasses = 2;
const string  suitNames[] = { "Spade",  "Club", "Heart", "Diamond"};
const string rankNames[] = {"Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace" };

Mat trainX, trainY;
vector<int> trainLabels;

//Function getCardColor returns color of the card(black - 0; red - 1)
int getCardColor(const Mat& cardImg)
{
	const int threshold = 50;
	int goodPixelsAmount = 0;
	for (int y = 0; y < cardImg.size().height; y++)
	{
		for (int x = 0; x < cardImg.size().width; x++)
		{
			Vec3b currentPixel = cardImg.at<Vec3b>(y, x);
			if ((currentPixel.val[2] - currentPixel.val[0] >= threshold) && (currentPixel.val[2] - currentPixel.val[1] >= threshold))
			{
				goodPixelsAmount++;
			}
		}
	}

	return static_cast<int>((1.0 * goodPixelsAmount / cardImg.size().area()) > 0.05);
}

/*
int getCardColor(const Mat& cardImg)
{
	const int hueThreshold = 125;
	int goodPixelsAmount = 0;
	Mat hsvImg = cardImg.clone();
	cvtColor(hsvImg, hsvImg, CV_BGR2HSV);

	for (int y = 0; y < hsvImg.size().height; y++)
	{
		for (int x = 0; x < hsvImg.size().width; x++)
		{
			Vec3b currentPixel = hsvImg.at<Vec3b>(y, x);
			if (currentPixel.val[0] >= hueThreshold)
			{
				goodPixelsAmount++;
			}
		}
	}

	return static_cast<int>((1.0 * goodPixelsAmount / cardImg.size().area()) > 0.07);
}

*/

void preprocessImage(Mat &img, const Size &sz)
{
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY);
	threshold(grayImg, grayImg, 100, 255, CV_THRESH_BINARY_INV);
	resize(grayImg, grayImg, Size(sz.width, sz.height));
	img = grayImg.clone();
}


void preprocessRankImage(Mat &rank)
{
	Mat grayImg;
	cvtColor(rank, grayImg, CV_BGR2GRAY);
	threshold(grayImg, grayImg, 100, 255, CV_THRESH_BINARY_INV);
	resize(grayImg, grayImg, Size(scaledRankWidth, scaledRankHeight));
	rank = grayImg.clone();
}


void preprocessSuitImage(Mat &suit)
{
	Mat grayImg;
	cvtColor(suit, grayImg, CV_BGR2GRAY);
	threshold(grayImg, grayImg, 100, 255, CV_THRESH_BINARY_INV);
	resize(grayImg, grayImg, Size(scaledSuitWidth, scaledSuitHeight));
	suit = grayImg.clone();
}

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

Mat generateImg(Mat image, bool rotation, bool shift, int angle = 0, int maxShiftX = 0, int maxShiftY = 0)
{
	//rotate image
	if (rotation)
	{
		double angleValue = rand() % (angle + 1);
		if (rand() % 2 > 0) angleValue *= -1.0;
		Mat rotationMatrix2D = getRotationMatrix2D(Point2f((image.rows - 1) / 2, (image.cols - 1) / 2), angleValue, 1.0);
		warpAffine(image, image, rotationMatrix2D, image.size());
	}

	if (shift)
	{
		int shiftX = rand() % (maxShiftX + 1);
		int shiftY = rand() % (maxShiftY + 1);
		Point2f srcTri[3];
		Point2f dstTri[3];

		srcTri[0] = Point2f(0, 0);
		srcTri[1] = Point2f(image.cols - 1.f, 0);
		srcTri[2] = Point2f(0, image.rows - 1.f);

		dstTri[0] = Point2f(0 + shiftX, 0 + shiftY);
		dstTri[1] = Point2f(image.cols - 1.f + shiftX, 0 + shiftY);
		dstTri[2] = Point2f(0 + shiftX, image.rows - 1.f + shiftY);

		Mat warpMat = getAffineTransform(srcTri, dstTri);
		warpAffine(image, image, warpMat, image.size());
	}

	return image;
}

Mat formImageData(const Mat& image)
{

	Mat result = Mat::zeros(Size(image.size().area(), 1), CV_32FC1);
	int imgWidth = image.size().width;
	for (int y = 0; y < image.size().height; y++)
	{
		for (int x = 0; x < image.size().width; x++)
		{
			*result.ptr<float>(0, y * imgWidth + x) = *image.ptr<uchar>(y, x);
		}
	}
	return result;
}

vector<Mat> generateImgKNN(Mat& image, int nInstances) {
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

int findFrequentClass(vector<int> v)
{
	sort(v.begin(), v.end());
	int counter = 1;
	pair<int, int> answer = { 1, v[0] };
	for (int i = 1; i < (int)v.size(); i++)
	{
		if (v[i] == v[i - 1])
		{
			counter++;
		}
		else
		{
			counter = 1;
		}

		if (counter > answer.first)
		{
			answer = { counter, v[i] };
		}
	}
	return answer.second;
}

int main()
{
	Mat suitsCoverImg[4], suitsCoverMask[4], ranksCoverImg[5], ranksCoverMask[5];

	suitsCoverImg[0] = imread("./images/spade.jpg");
	suitsCoverImg[1] = imread("./images/club.png");
	suitsCoverImg[2] = imread("./images/heart.png");
	suitsCoverImg[3] = imread("./images/diamond.jpg");

	suitsCoverMask[0] = imread("./images/spade_mask.jpg");
	suitsCoverMask[1] = imread("./images/club_mask.png");
	suitsCoverMask[2] = imread("./images/heart_mask.png");
	suitsCoverMask[3] = imread("./images/diamond_mask.jpg");

	ranksCoverImg[0] = imread("./images/Ten.jpg");
	ranksCoverImg[1] = imread("./images/Jack.jpg");
	ranksCoverImg[2] = imread("./images/Queen.jpg");
	ranksCoverImg[3] = imread("./images/King.jpg");
	ranksCoverImg[4] = imread("./images/Ace.jpg");

	ranksCoverMask[0] = imread("./images/Ten_mask.jpg");
	ranksCoverMask[1] = imread("./images/Jack_mask.jpg");
	ranksCoverMask[2] = imread("./images/Queen_mask.jpg");
	ranksCoverMask[3] = imread("./images/King_mask.jpg");
	ranksCoverMask[4] = imread("./images/Ace_mask.jpg");


	for (int idx = 0; idx < 4; idx++)
	{
		resize(suitsCoverImg[idx], suitsCoverImg[idx], Size(cardWidth, cardHeight));
		resize(suitsCoverMask[idx], suitsCoverMask[idx], Size(cardWidth, cardHeight));

	}

	for (int idx = 0; idx < 5; idx++)
	{
		resize(ranksCoverMask[idx], ranksCoverMask[idx], Size(cardWidth, cardHeight));
		resize(ranksCoverImg[idx], ranksCoverImg[idx], Size(cardWidth, cardHeight));
	}

	FileStorage ffsBlackCard("MLPBlackSuits.xml", FileStorage::READ);
	Ptr<cv::ml::ANN_MLP> ANNBlackCard = cv::Algorithm::read<cv::ml::ANN_MLP>(ffsBlackCard.root());

	FileStorage ffsRedCard("MLPRedSuits.xml", FileStorage::READ);
	Ptr<cv::ml::ANN_MLP> ANNRedCard = cv::Algorithm::read<cv::ml::ANN_MLP>(ffsRedCard.root());

	FileStorage ffsRank("MLPRank.xml", FileStorage::READ);
	Ptr<cv::ml::ANN_MLP> ANNCardRank = cv::Algorithm::read<cv::ml::ANN_MLP>(ffsRank.root());

	FileStorage ffsKNNSuit("KNNSuits.xml", FileStorage::READ);
	Ptr<cv::ml::KNearest> KNNCardSuit = cv::ml::StatModel::load<cv::ml::KNearest>("KNNSuits.xml");

	VideoCapture cap(1);
	Mat img, cameraImg;
	while (waitKey(30) != 'e')
	{
		if (waitKey(500) == 's')
		{
			cap >> img;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			findCardsContours(img, contours, hierarchy);

			////Drawing contours
			//Mat result = img.clone();
			//int idx = 0;
			//for (; idx >= 0; idx = hierarchy[idx][0])
			//{
			//	drawContours(result, contours, idx, Scalar(0, 255, 0), 5, 8, hierarchy);
			//}

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
					double vectorsMult = (approxVector[2].x - approxVector[0].x) * (approxVector[3].x - approxVector[1].x) + (approxVector[3].y - approxVector[1].y) * (approxVector[2].y - approxVector[0].y);
					
					if (vectorsMult > 0.0)
					{
						swap(approxVector[0], approxVector[1]);
						swap(approxVector[1], approxVector[2]);
						swap(approxVector[2], approxVector[3]);
					}

					Point2f dstPTPoints[4], srcPTPoints[4];

					for (int i = 0; i < 4; i++)
					{
						circle(cImg, approxVector[i], 5, color[i], 2);
						srcPTPoints[i] = approxVector[i];
					}

					int dist1 = getDistance(approxVector[0], approxVector[1]);
					int dist2 = getDistance(approxVector[1], approxVector[2]);

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
			
			vector<Mat> suitsImg;
			vector<Mat> ranksImg;

			Mat currentCard;
			for (int idx = 0; idx < cards.size(); idx++)
			{
				currentCard = cards[idx].clone();
				ranksImg.push_back(currentCard(Rect(0, 0, originalRankWidth, originalRankHeight)));
				suitsImg.push_back(currentCard(Rect(0, originalRankHeight, origialSuitWidth, originalSuitHeight)));
			}
			
			//Mat suitsData = Mat::zeros(Size(scaledSuitWidth * scaledSuitHeight, suitsImg.size()), CV_32FC1);
			//Mat ranksData = Mat::zeros(Size(scaledRankWidth * scaledRankHeight, ranksImg.size()), CV_32FC1);

			Mat suitsConcat;
			vector<int> cardsColors;
			vector<Mat> suitDescriptors[5];
			for (int idx = 0; idx < suitsImg.size(); idx++)
			{
				cardsColors.push_back(getCardColor(suitsImg[idx]));
				vector<Mat> generatedImgKNN = generateImgKNN(suitsImg[idx], 5);
				for (int idxDescriptor = 0; idxDescriptor < 5; idxDescriptor++)
				{
						suitDescriptors[idx].push_back(Features_Mapper::prepare_descriptors(generatedImgKNN[idxDescriptor], 0));
				}
				
				preprocessImage(suitsImg[idx], Size(scaledSuitWidth, scaledSuitHeight));
				preprocessImage(ranksImg[idx], Size(scaledRankWidth, scaledRankHeight));

				if (idx == 0)
				{
					suitsConcat = suitsImg[idx].clone();
				}
				else
				{
					hconcat(vector<Mat>{suitsConcat, suitsImg[idx]}, suitsConcat);
				}

				/*for (int y = 0; y < scaledSuitHeight; y++)
				{
					for (int x = 0; x < scaledSuitWidth; x++)
					{
						*suitsData.ptr<float>(idx, y * scaledSuitWidth + x) = *suitsImg[idx].ptr<uchar>(y, x);
					}
				}

				for (int y = 0; y < scaledRankHeight; y++)
				{
					for (int x = 0; x < scaledRankWidth; x++)
					{
						*ranksData.ptr<float>(idx, y * scaledRankWidth + x) = *ranksImg[idx].ptr<uchar>(y, x);
					}
				}*/
			}

			
			for (int idx = 0; idx < suitsImg.size(); idx++)
			{
				vector<int> suitPredictions;

				int ANNPredictedSuit;
				if (cardsColors[idx] == 0)
				{
					Mat generatedImg = suitsImg[idx].clone();
					ANNPredictedSuit = ANNBlackCard->predict(formImageData(generatedImg), noArray());
					suitPredictions.push_back(ANNPredictedSuit);
					for (int genIt = 0; genIt < 4; genIt++)
					{
						generatedImg = generateImg(suitsImg[idx], true, true, 5, 2, 2);
						ANNPredictedSuit = ANNBlackCard->predict(formImageData(generatedImg), noArray());
						suitPredictions.push_back(ANNPredictedSuit);
					}
				}
				else
				{
					Mat generatedImg = suitsImg[idx].clone();
					ANNPredictedSuit = ANNRedCard->predict(formImageData(generatedImg), noArray());
					suitPredictions.push_back(ANNPredictedSuit + 2);
					for (int genIt = 0; genIt < 4; genIt++)
					{
						generatedImg = generateImg(suitsImg[idx], true, true, 5, 2, 2);
						ANNPredictedSuit = ANNRedCard->predict(formImageData(generatedImg), noArray());
						suitPredictions.push_back(ANNPredictedSuit + 2);
					}
				}
				
				for (int ii = 0; ii < 5; ii++)
				{
					Mat KNNSuitResult;
					KNNCardSuit->findNearest(suitDescriptors[idx][ii], 60, KNNSuitResult);
					int KNNPredictedSuit = static_cast<int>(KNNSuitResult.at<float>(0, 0));
					if ((cardsColors[idx] == 0 && KNNPredictedSuit < 2) || (cardsColors[idx] == 1 && KNNPredictedSuit > 1))
					{
						suitPredictions.push_back(KNNPredictedSuit);
					}
				}
				
				for (int k = 0; k < suitPredictions.size(); k++)
				{
					cout << suitPredictions[k] << " ";
				}

				cout << endl;

				int predictedSuit = findFrequentClass(suitPredictions);
				int predictedRank = ANNCardRank->predict(formImageData(ranksImg[idx]), noArray());

				suitsCoverImg[predictedSuit].copyTo(cards[idx], suitsCoverMask[predictedSuit]);
				ranksCoverImg[predictedRank].copyTo(cards[idx], ranksCoverMask[predictedRank]);
				cout << "Predicted suit for card #" << idx + 1 << ": "  << predictedSuit << endl;
				cout << "Predicted rank for card #" << idx + 1 << ": " << rankNames[predictedRank] << endl;
			}

			cout << "==================" << endl;
			
			Mat result = cards[0].clone();
			for (int idx = 1; idx < cards.size(); idx++)
			{
				Mat currentCard = cards[idx].clone();
				hconcat(vector<Mat>{result, currentCard}, result);
			}

			resize(cImg, cImg, Size(result.size().width, 0.5 * result.size().width * cImg.size().height / cImg.size().width));
			vconcat(vector<Mat>{cImg, result}, result);
			cv::imshow("Result", result);
		}
		cap >> cameraImg;
		cv::imshow("Camera", cameraImg);	
	}
	
	cv::waitKey();
}