#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

#include <opencv2/opencv.hpp>


int main(int argc, char *argv[]){
	//extract20HsvFeatures();
	extract60ColorThemes();
	//extract80HsvFeatures();
	extract768ColorHistogram("bgr");
	extract768ColorHistogram("hsv");
	extract768ColorHistogram("lab");
	//extract4608HsvFeatures();

	system("pause");
	return 0;

}