#pragma once
#include <iostream>


class Samples_Generator
{

private:
	std::string imgDir;
	std::string datasetDir;
	
	int nInstances;

public:

	Samples_Generator(std::string imageDirectory, std::string datasetDirectory, int n, int imgSize);

	~Samples_Generator();
};