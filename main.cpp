#include "accessDirectory.h"
#include "functions.h"

#include <opencv2/opencv.hpp>


int main(int argc, char *argv[])
{
	//extract20HsvFeatures();
	//extract80HsvFeatures();
	extract768ColorHistogram("bgr");
	extract768ColorHistogram("hsv");
	extract768ColorHistogram("lab");
	extract768BgrFeatures();
	extract768HsvFeatures();
	extract768LabFeatures();
	//extract4608HsvFeatures();


	system("pause");
	return 0;

}