#include "accessDirectory.h"
#include "functions.h"

using namespace std;

void extract768HsvFeatures() {

	for (int entireItr = 0; entireItr < 2; entireItr++) {

		bool isTrainData = true;

		if (entireItr == 0) {
			isTrainData = true;
		}
		else {
			isTrainData = false;
		}

		string rootDir = "";
		string toDir = "";
		if (isTrainData) {
			rootDir = "C:/MUSIC_RECOMMENDATION/src_data/shots_OMV200/";
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV200_csv_shot_768hsv/";
		}
		else {
			rootDir = "C:/MUSIC_RECOMMENDATION/src_data/shots_recommendation_test/";
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/recommendation_test_features/csv_shot_768hsv/";
		}

		vector<string> videoList = Dir::readIncludingFolder(rootDir);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> frameList = Dir::readOutOfFolder(rootDir + videoList[videoItr]);

				vector<vector<int>> hsvFeatures;

				// 各フレームの取得
				for (int frameItr = 0; frameItr < frameList.size(); frameItr++) {
					if (frameList[frameItr] != "." && frameList[frameItr] != "..") {

						string srcPath = rootDir + videoList[videoItr] + "/" + frameList[frameItr];

						cv::Mat uSrc = cv::imread(srcPath);
						cv::resize(uSrc, uSrc, cv::Size(), 256 / (double)uSrc.cols, 256 / (double)uSrc.rows);

						cv::Mat uHsv;
						cv::cvtColor(uSrc, uHsv, CV_BGR2HSV_FULL);

						// 各ピクセルのHSVの値を出力
						cv::Mat uChannels[3];
						cv::split(uHsv, uChannels);

						vector<vector<int>> iHsvHist(3, vector<int>(256, 0));

						vector<int> tmpHsvFeatures;

						//各フレームのHSV値をそれぞれ256ビンのヒストグラムにする
						for (int y = 0; y < uHsv.rows; y++) {
							for (int x = 0; x < uHsv.cols; x++) {

								int hue = (int)uChannels[0].ptr<unsigned char>(y)[x];
								int sat = (int)uChannels[1].ptr<unsigned char>(y)[x];
								int val = (int)uChannels[2].ptr<unsigned char>(y)[x];

								iHsvHist[0][hue] += 1;
								iHsvHist[1][sat] += 1;
								iHsvHist[2][val] += 1;
							}
						}

						for (int hsvItr = 0; hsvItr < 3; hsvItr++) {
							for (int chItr = 0; chItr < 256; chItr++) {
								tmpHsvFeatures.push_back(iHsvHist[hsvItr][chItr]);
							}
						}

						hsvFeatures.push_back(tmpHsvFeatures);
					}
				}

				cout << hsvFeatures.size() << endl;
				cout << hsvFeatures.front().size() << endl;

				// csvで出力
				string csv_name = toDir + videoList[videoItr] + "_shot_hsv768.csv";

				ofstream ofs(csv_name);
				for (int dataItr = 0; dataItr < hsvFeatures.size(); dataItr++) {
					for (int featureItr = 0; featureItr < hsvFeatures.front().size(); featureItr++) {

						if (featureItr == hsvFeatures.front().size() - 1) {
							ofs << hsvFeatures[dataItr][featureItr] << endl;
						}
						else {
							ofs << hsvFeatures[dataItr][featureItr] << ",";
						}
					}
				}
			}
		}
	}
}