#include "accessDirectory.h"
#include "functions.h"

using namespace std;

void extract768BgrFeatures() {

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
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV200_csv_shot_768bgr/";
		}
		else {
			rootDir = "C:/MUSIC_RECOMMENDATION/src_data/shots_recommendation_test/";
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/recommendation_test_features/csv_shot_768bgr/";
		}

		vector<string> videoList = Dir::readIncludingFolder(rootDir);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> frameList = Dir::readOutOfFolder(rootDir + videoList[videoItr]);

				vector<vector<int>> bgrFeatures;

				// 各フレームの取得
				for (int frameItr = 0; frameItr < frameList.size(); frameItr++) {
					if (frameList[frameItr] != "." && frameList[frameItr] != "..") {

						string srcPath = rootDir + videoList[videoItr] + "/" + frameList[frameItr];

						cv::Mat uBgr = cv::imread(srcPath);
						cv::resize(uBgr, uBgr, cv::Size(), 256 / (double)uBgr.cols, 256 / (double)uBgr.rows);

						// 各ピクセルのhsvの値を出力
						cv::Mat uChannels[3];
						cv::split(uBgr, uChannels);

						vector<vector<int>> iBgrHist(3, vector<int>(256, 0));

						vector<int> tmpBgrFeatures;

						//各フレームのRGB値をそれぞれ256ビンのヒストグラムにする
						for (int y = 0; y < uBgr.rows; y++) {
							for (int x = 0; x < uBgr.cols; x++) {

								int blue = (int)uChannels[0].ptr<unsigned char>(y)[x];
								int green = (int)uChannels[1].ptr<unsigned char>(y)[x];
								int red = (int)uChannels[2].ptr<unsigned char>(y)[x];

								iBgrHist[0][blue] += 1;
								iBgrHist[1][green] += 1;
								iBgrHist[2][red] += 1;
							}
						}

						for (int bgrItr = 0; bgrItr < 3; bgrItr++) {
							for (int chItr = 0; chItr < 256; chItr++) {
								tmpBgrFeatures.push_back(iBgrHist[bgrItr][chItr]);
							}
						}

						bgrFeatures.push_back(tmpBgrFeatures);
					}
				}

				cout << bgrFeatures.size() << endl;
				cout << bgrFeatures.front().size() << endl;

				// csvで出力
				string csv_name = toDir + videoList[videoItr] + "_shot_bgr768.csv";

				ofstream ofs(csv_name);
				for (int dataItr = 0; dataItr < bgrFeatures.size(); dataItr++) {
					for (int featureItr = 0; featureItr < bgrFeatures.front().size(); featureItr++) {

						if (featureItr == bgrFeatures.front().size() - 1) {
							ofs << bgrFeatures[dataItr][featureItr] << endl;
						}
						else {
							ofs << bgrFeatures[dataItr][featureItr] << ",";
						}
					}
				}
			}
		}
	}
}