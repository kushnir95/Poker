#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const string imgPath = "./Images/five/";
const string trainsetSuitsDir = "./trainset/suits";
const string trainsetRanksDir = "./trainset/ranks";
const int cardWidth = 200;
const int cardHeight = 300;
const int origialSuitWidth = 40;
const int originalSuitHeight = 35;
const int scaledSuitWidth = 20;
const int scaledSuitHeight = 20;

const int originalRankWidth = 35;
const int originalRankHeight = 60;
const int scaledRankWidth = 20;
const int scaledRankHeight = 40;
const int nRank = 4;

const int nclasses = 2;
const string  suitNames[] = { "Spade",  "Club", "Heart", "Diamond"};
const string rankNames[] = { "Jack", "Queen", "King", "Ace" };

Mat trainX, trainY;
vector<int> trainLabels;

//Function getCardColor returns color of the card(black - 0; red - 1)
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

	return static_cast<int>((1.0 * goodPixelsAmount / cardImg.size().area()) > 0.15);
}

void readData(const string& dir, const vector< pair<string, int> >& dirInfo, Mat& features, Mat& targets, vector<int>& labels)
{
	vector<Mat> images;

	labels.clear();
	for (int i = 0; i < dirInfo.size(); i++)
	{
		for (int idx = 1; idx <= dirInfo[i].second; idx++)
		{
			string imgPath = dir + "/" + dirInfo[i].first + "/" + to_string(idx) + ".jpg";
			Mat img = imread(imgPath);
			Mat grayImg;
			cvtColor(img, grayImg, CV_BGR2GRAY);
			images.push_back(grayImg);
			labels.push_back(i);
		}
	}

	features = Mat::zeros(Size(scaledSuitWidth * scaledSuitHeight, images.size()), CV_32FC1);
	targets = Mat::zeros(Size(nclasses, images.size()), CV_32FC1);

	int idx = 0;
	for (Mat currentInstance : images)
	{
		for (int y = 0; y < scaledSuitHeight; y++)
		{
			for (int x = 0; x < scaledSuitWidth; x++)
			{
				features.at<float>(idx, y * scaledSuitWidth + x) = currentInstance.at<uchar>(y, x);
			}
		}

		targets.at<float>(idx, labels[idx]) = 1.0;
		idx++;
	}
}

void readRankTrainset(const string& dir, const vector< pair<string, int> >& dirInfo, Mat& features, Mat& targets, vector<int>& labels)
{
	vector<Mat> images;

	labels.clear();
	for (int i = 0; i < dirInfo.size(); i++)
	{
		for (int idx = 1; idx <= dirInfo[i].second; idx++)
		{
			string imgPath = dir + "/" + dirInfo[i].first + "/" + to_string(idx) + ".jpg";
			Mat img = imread(imgPath);
			Mat grayImg;
			cvtColor(img, grayImg, CV_BGR2GRAY);
			images.push_back(grayImg);
			labels.push_back(i);
		}
	}

	features = Mat::zeros(Size(scaledRankWidth * scaledRankHeight, images.size()), CV_32FC1);
	targets = Mat::zeros(Size(nRank, images.size()), CV_32FC1);

	int idx = 0;
	for (Mat currentInstance : images)
	{
		for (int y = 0; y < scaledRankHeight; y++)
		{
			for (int x = 0; x < scaledRankWidth; x++)
			{
				features.at<float>(idx, y * scaledRankWidth + x) = currentInstance.at<uchar>(y, x);
			}
		}

		targets.at<float>(idx, labels[idx]) = 1.0;
		idx++;
	}
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


void extractCards(const Mat& src, vector<Mat>& cards)
{

	cards.clear();

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findCardsContours(src, contours, hierarchy);

	//Drawing contours

	//Mat result = img.clone();
	//int idx = 0;
	//for (; idx >= 0; idx = hierarchy[idx][0])
	//{
	//	drawContours(result, contours, idx, Scalar(0, 255, 0), 5, 8, hierarchy);
	//}

	for (int idx = 0; idx < contours.size(); idx++)
	{
		vector<Point> approxVector;
		approxPolyDP(contours[idx], approxVector, 15.0, true);

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
			warpPerspective(src, extractedCardImg, PTMatrix, extractedCardImg.size());

			cards.push_back(extractedCardImg);
		}
	}

}

int main()
{
	Mat suitsCoverImg[4], suitsCoverMask[4];

	suitsCoverImg[0] = imread("./images/spade.jpg");
	suitsCoverImg[1] = imread("./images/club.png");
	suitsCoverImg[2] = imread("./images/heart.png");
	suitsCoverImg[3] = imread("./images/diamond.jpg");

	suitsCoverMask[0] = imread("./images/spade_mask.jpg");
	suitsCoverMask[1] = imread("./images/club_mask.png");
	suitsCoverMask[2] = imread("./images/heart_mask.png");
	suitsCoverMask[3] = imread("./images/diamond_mask.jpg");

	for (int idx = 0; idx < 4; idx++)
	{
		resize(suitsCoverImg[idx], suitsCoverImg[idx], Size(cardWidth, cardHeight));
		resize(suitsCoverMask[idx], suitsCoverMask[idx], Size(cardWidth, cardHeight));
	}

	//Creating neural network for recognizing suits

	vector<pair<string, int>> trainDirInfo = {
		{ "1", 1000 },
		{ "2", 1000 }
	};

	cout << "Reading train data(1)..." << endl;
	readData(trainsetSuitsDir, trainDirInfo, trainX, trainY, trainLabels);
	
	Ptr<cv::ml::ANN_MLP> ANNBlackCard = cv::ml::ANN_MLP::create();
	Mat_<int> layers(5, 1);
	layers(0) = scaledSuitWidth * scaledSuitHeight;
	layers(1) = 300;
	layers(2) = 100;
	layers(3) = 50;
	layers(4) = nclasses;

	ANNBlackCard->setLayerSizes(layers);
	ANNBlackCard->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1.0, 0.0);
	ANNBlackCard->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
	ANNBlackCard->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

	std::cout << "Start training network(1)..." << std::endl;

	ANNBlackCard->train(trainX, ml::ROW_SAMPLE, trainY);

	std::cout << "Finish training network(1)..." << std::endl;

	trainDirInfo = 
	{
		{"3", 1000},
		{"4", 1000}
	};

	cout << "Reading train data(2)..." << endl;
	readData(trainsetSuitsDir, trainDirInfo, trainX, trainY, trainLabels);

	Ptr<cv::ml::ANN_MLP> ANNRedCard = cv::ml::ANN_MLP::create();

	ANNRedCard->setLayerSizes(layers);
	ANNRedCard->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1.0, 0.0);
	ANNRedCard->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
	ANNRedCard->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

	std::cout << "Start training network(2)..." << std::endl;

	ANNRedCard->train(trainX, ml::ROW_SAMPLE, trainY);

	std::cout << "Finish training network(2)..." << std::endl;


	
	cout << "Reading train data(3)..." << endl;

	trainDirInfo = {
		{"1", 1000},
		{"2", 1000},
		{"3", 1000},
		{"4", 1000}
	};

	readRankTrainset(trainsetRanksDir, trainDirInfo, trainX, trainY, trainLabels);

	Ptr<cv::ml::ANN_MLP> ANNCardRank = cv::ml::ANN_MLP::create();
	Mat_<int> layers2(5, 1);
	layers2(0) = scaledRankWidth * scaledRankHeight;
	layers2(1) = 300;
	layers2(2) = 100;
	layers2(3) = 50;
	layers2(4) = nRank;

	ANNCardRank->setLayerSizes(layers2);
	ANNCardRank->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1.0, 0.0);
	ANNCardRank->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 0.0001));
	ANNCardRank->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

	std::cout << "Start training network(3)..." << std::endl;

	ANNCardRank->train(trainX, ml::ROW_SAMPLE, trainY);

	std::cout << "Finish training network(3)..." << std::endl;
	
	VideoCapture cap(1);
	Mat img, cameraImg;
	while (waitKey(30) != 'e')
	{
		if (waitKey(500) == 's')
		{
			std::cout << "Processing image" << std::endl;
			cap >> img;
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
			
			Mat suitsData = Mat::zeros(Size(scaledSuitWidth * scaledSuitHeight, suitsImg.size()), CV_32FC1);
			Mat ranksData = Mat::zeros(Size(scaledRankWidth * scaledRankHeight, ranksImg.size()), CV_32FC1);

			Mat suitsConcat;
			vector<int> cardsColors;
			for (int idx = 0; idx < suitsImg.size(); idx++)
			{
				cardsColors.push_back(getCardColor(suitsImg[idx]));
				preprocessSuitImage(suitsImg[idx]);
				preprocessRankImage(ranksImg[idx]);

				if (idx == 0)
				{
					suitsConcat = suitsImg[idx].clone();
				}
				else
				{
					hconcat(vector<Mat>{suitsConcat, suitsImg[idx]}, suitsConcat);
				}

				for (int y = 0; y < scaledSuitHeight; y++)
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
				}
			}

			//cv::imshow("Suits", suitsConcat);

			
			for (int idx = 0; idx < suitsImg.size(); idx++)
			{
				int predictedSuit;
				if (cardsColors[idx] == 0)
				{
					predictedSuit = ANNBlackCard->predict(suitsData.row(idx), noArray());
				}
				else
				{
					predictedSuit = ANNRedCard->predict(suitsData.row(idx), noArray()) + 2;
				}
				int predictedRank = ANNCardRank->predict(ranksData.row(idx), noArray());

				suitsCoverImg[predictedSuit].copyTo(cards[idx], suitsCoverMask[predictedSuit]);
				cout << "Predicted suit for card #" << idx + 1 << " "  << predictedSuit  << endl;
				cout << "Predicted rank for card #" << idx + 1 << " " << rankNames[predictedRank] << endl;
			}

			result = cards[0].clone();
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