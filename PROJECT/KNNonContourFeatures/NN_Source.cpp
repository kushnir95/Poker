#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Samples_Generator.h"				// � ����� ����� ���� ������� .h i .cpp
#include "Features_Mapper.h"

using namespace std;
using namespace cv;

string imageDirectory = "./images";			// ������ ����� ��� ����������
string datasetDirectory = "./trainset";		// ������� ����� ��� ���������� � ������ ��� Features_Mapper

string dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND",
		"TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
		"NINE", "TEN", "JACK", "QUEEN", "KING", "ACE" };

/*
K-Nearest Neighbors model
ѳ��� ����������� � ����� ��� ������ � Features_Mapper: ���� float �������, �� �������������� ������.
���������:
	samples - ������� ������ ����������� ��� ����������;
	neib - ������� ���������� �����, ��������������� � ������� findNearest(), ��� ����������� ���������;
	isSuits - ������, ���� �����, �� ���� ���� ��������� ������� (4 �����) �� ������� (13 �����).
*/
void KNN(int samples, int neib,  int isSuits) {
	int startIdx = 0;
	int endIdx = 0;
	
	if (isSuits) {
		endIdx = 4;
	}
	else {
		startIdx = 4;
		endIdx = 17;
	}

	vector<Mat> trainSet;
	vector<Mat> trainSet_responses;

	/*
	���������� �� ����� dirInfo, ������ ������ ������������ �� ��������� ���������.
	startIdx �� endIdx - �������� �� �������� ������ ����.
	��� startIdx = 0; endIdx = 4 ������ ���� dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND" }
	��� ����� startIdx = 4; endIdx = 17;
	*/
	for (int i = startIdx; i < endIdx; i++) {
		/*
		� ������ trainSet ����������� ������� ������ featureNumber x samples, 
		�� ���� - ����������� ������� ��������� ������.
		���������� ����������� �������� Features_Mapper::prepare_descriptors()

		� ������ trainSet_responses ����������� ������� 1 x samples (����������),
		�� ������� ������ ������� ����� �������� int, �� ���� ��������������� ������ � ����� dirInfo
		*/
		trainSet.push_back(Features_Mapper::prepare_descriptors(datasetDirectory + "/" + dirInfo[i], samples));
		Mat responses = Mat::zeros(Size(1, samples), CV_32SC1);
		for (int y = 0; y < samples; y++) {
			responses.at<int>(y, 0) = i;
		}
		trainSet_responses.push_back(responses);
	}
	Mat TS, TS_responses;
	vconcat(trainSet, TS);
	vconcat(trainSet_responses, TS_responses);
	/*
	�������� �������� ������� � �� Y ������� �'������� �������� ������� trainSet �� trainSet_responses �� ��������.
	� ��������� ������� TS i TS_responses ����� ������ samples x classNumber
	*/

	Ptr<ml::KNearest> KNN = ml::KNearest::create();		// ����������� ������� ������ ���

	cout << "Start training network..." << endl;
	KNN->train(ml::TrainData::create(TS, ml::ROW_SAMPLE, TS_responses));	// ������ ��������� �� ������ � TS, �������������� ������ � TS_responses;
	cout << "Finish training network..." << endl;

	/*
	�������� ���������� ����������.
	string imgPath - ���������, �� ������ ���������� ��� ����������.
	� ��� ��������� �������� Features_Mapper::prepare_descriptors() ����������� �����������.

	int imgCount - ������� ���������, �� ������ � �������� (��������� ���� ������)
	Mat prediction - ������� ����������, �� ������ float �������� (�� ��� ������� �����). 
	���� ����� �������� �� int, ��� ����������� � ����� ������� ��� dirInfo
	*/
	int imgCount = 65;
	string imgPath = "./img_predict/";
	Mat image_desc = Features_Mapper::prepare_descriptors(imgPath, imgCount);
	Mat prediction;
	KNN->findNearest(image_desc, neib, prediction);
	float u = prediction.at<float>(0, 0);

	if (isSuits) {
		cout << "KNN for SUITS prediction:" << endl;
	} else cout << "KNN for RANK prediction:" << endl;
	for (int i = 0; i < imgCount; i++) {
		cout <<  i << ".	" << dirInfo[(int)prediction.at<float>(i, 0)] << endl;
	}
	cout << endl;
}

int main() {
	
	/*
	������ sampleCount = 400 ������ ��� �������� ������ ����.
	�������� ������ ��������� (� ���� ����������) � ������� ��������� (���� �������� ����������)
	�������� ����� ����������� ��������� imgSize = 51
	
	�� ������� ����� ���������������, ���� ��������� ./trainset �������.
	*/
	//Samples_Generator SG(imageDirectory, datasetDirectory, 400, 51);

	/*
	����������� ������ ��� ��� ������.
	�-��� ������ = 40;
	�-��� ���������� ����� � ������� findNearest() = 10
	*/
	KNN(40, 10, 1);

	/*
	����������� ������ ��� ��� �����.
	�-��� ������ = 280;
	�-��� ���������� ����� � ������� findNearest() = 35
	*/
	KNN(280, 35, 0);


return 0;
}