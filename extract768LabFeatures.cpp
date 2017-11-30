#include "accessDirectory.h"
#include "functions.h"

using namespace std;


void extract768LabFeatures() {

	for (int entireItr = 0; entireItr < 1; entireItr++) {

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
			rootDir = "../../src_data/shots_OMV200/";
			toDir = "../../src_data/train_features/OMV200_csv_shot_768lab/";
		}
		else {
			rootDir = "../../src_data/shots_recommendation_test/";
			toDir = "../../src_data/recommendation_test_features/csv_shot_768lab/";
		}

		vector<string> videoList = Dir::readIncludingFolder(rootDir);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> frameList = Dir::readOutOfFolder(rootDir + videoList[videoItr]);

				vector<vector<int>> labFeatures;

				// 各フレームの取得
				for (int frameItr = 0; frameItr < frameList.size(); frameItr++) {
					if (frameList[frameItr] != "." && frameList[frameItr] != "..") {

						string srcPath = rootDir + videoList[videoItr] + "/" + frameList[frameItr];

						cv::Mat uSrc = cv::imread(srcPath);
						cv::resize(uSrc, uSrc, cv::Size(), 256 / (double)uSrc.cols, 256 / (double)uSrc.rows);

						cv::Mat uLab;
						cv::cvtColor(uSrc, uLab, CV_BGR2Lab);

						// 各ピクセルのLabの値を出力
						cv::Mat uChannels[3];
						cv::split(uLab, uChannels);

						vector<vector<int>> iLabHist(3, vector<int>(256, 0));

						vector<int> tmpLabFeatures;

						//各フレームのLab値をそれぞれ256ビンのヒストグラムにする
						for (int y = 0; y < uLab.rows; y++) {
							for (int x = 0; x < uLab.cols; x++) {

								int lValue = (int)uChannels[0].ptr<unsigned char>(y)[x];
								int aValue = (int)uChannels[1].ptr<unsigned char>(y)[x];
								int bValue = (int)uChannels[2].ptr<unsigned char>(y)[x];

								iLabHist[0][lValue] += 1;
								iLabHist[1][aValue] += 1;
								iLabHist[2][bValue] += 1;
							}
						}

						for (int labItr = 0; labItr < 3; labItr++) {
							for (int chItr = 0; chItr < 256; chItr++) {
								tmpLabFeatures.push_back(iLabHist[labItr][chItr]);
							}
						}

						labFeatures.push_back(tmpLabFeatures);
					}
				}

				cout << labFeatures.size() << endl;
				cout << labFeatures.front().size() << endl;

				// csvで出力
				string csv_name = toDir + videoList[videoItr] + "_shot_lab768.csv";

				ofstream ofs(csv_name);
				for (int dataItr = 0; dataItr < labFeatures.size(); dataItr++) {
					for (int featureItr = 0; featureItr < labFeatures.front().size(); featureItr++) {

						if (featureItr == labFeatures.front().size() - 1) {
							ofs << labFeatures[dataItr][featureItr] << endl;
						}
						else {
							ofs << labFeatures[dataItr][featureItr] << ",";
						}
					}
				}
			}
		}
	}
}