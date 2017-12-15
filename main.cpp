#include "accessDirectory.h"
#include "extract_video_features_funcs.h"




int main(int argc, char *argv[]) {

	// extract20HsvFeatures();
	//extractColorThemes("bgr", "em");
	//extractColorThemes("hsv", "em");
	//extractColorThemes("lab", "em");
	// extract80HsvFeatures();

	//extract512ColorHistogram("bgr");
	//extract512ColorHistogram("hsv");
	//extract512ColorHistogram("lab");

	extract768ColorHistogram("bgr");
	extract768ColorHistogram("hsv");
	extract768ColorHistogram("lab");

	//extract4608HsvFeatures();

	return 0;
}