#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "Samples_Generator.h"				// У рішенні мають бути відповідні .h i .cpp
#include "Features_Mapper.h"

using namespace std;
using namespace cv;

string imageDirectory = "./images";			// Вхідна папка для генератора
string datasetDirectory = "./trainset";		// Вихідна папка для генератора і вхідна для Features_Mapper

string dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND",
		"TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
		"NINE", "TEN", "JACK", "QUEEN", "KING", "ACE" };

/*
K-Nearest Neighbors model
Сітка використовує в якості фіч аутпут з Features_Mapper: набір float значень, що характеризують контур.
Аргументи:
	samples - кількість семплів дескрипторів для тренування;
	neib - кількість найближчих сусідів, використовується у функції findNearest(), щоб передбачити результат;
	isSuits - тригер, який вказує, що сітка буде займатись мастями (4 класи) чи рангами (13 класів).
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
	Орієнтуючись на масив dirInfo, семпли будуть витягуватись із відповідних директорій.
	startIdx та endIdx - вказують які директорії будуть взяті.
	При startIdx = 0; endIdx = 4 будуть взяті dirInfo[] = { "SPADE", "CLUB", "HEART", "DIAMOND" }
	Для рангів startIdx = 4; endIdx = 17;
	*/
	for (int i = startIdx; i < endIdx; i++) {
		/*
		У вектор trainSet закидуються матриці розміру featureNumber x samples, 
		де фічі - дескриптори контурів відповідних семплів.
		Десриптори витягуються функцією Features_Mapper::prepare_descriptors()

		У вектор trainSet_responses закидуються матриці 1 x samples (вертикальні),
		де кожному семплу відповідає певне значення int, що буде характеризувати індекс в масиві dirInfo
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
	Будується загальна матриця Х та Y методом з'єднання елементів векторів trainSet та trainSet_responses по вертикалі.
	У результаті матриці TS i TS_responses мають висоту samples x classNumber
	*/

	Ptr<ml::KNearest> KNN = ml::KNearest::create();		// Створюється порожня модель КНН

	cout << "Start training network..." << endl;
	KNN->train(ml::TrainData::create(TS, ml::ROW_SAMPLE, TS_responses));	// Модель тренується на данних з TS, використовуючи відповіді в TS_responses;
	cout << "Finish training network..." << endl;

	/*
	Перевірка результатів тренування.
	string imgPath - директорія, де лежать зображення для тестування.
	З цих зображень функцією Features_Mapper::prepare_descriptors() витягуються дескриптори.

	int imgCount - кількість зображень, що лежить в директорії (задаються поки вручну)
	Mat prediction - матриця результатів, де лежать float значення (по суті індекси класів). 
	Потім треба привести до int, щоб використати в якості індекса для dirInfo
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
	Генерує sampleCount = 400 семплів для подальшої роботи сітки.
	Задається вхідна директорія (з чого генерувати) і вихідна директорія (куди складати генероване)
	Задається розмір генерованих зображень imgSize = 51
	
	Цю функцію треба використовувати, якщо директорія ./trainset порожня.
	*/
	//Samples_Generator SG(imageDirectory, datasetDirectory, 400, 51);

	/*
	Запускається модель КНН для мастей.
	К-сть семплів = 40;
	К-сть найближчих точок в функції findNearest() = 10
	*/
	KNN(40, 10, 1);

	/*
	Запускається модель КНН для рангів.
	К-сть семплів = 280;
	К-сть найближчих точок в функції findNearest() = 35
	*/
	KNN(280, 35, 0);


return 0;
}