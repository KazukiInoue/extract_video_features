#include <numeric>

#include "accessDirectory.h"
#include "functions.h"

using namespace std;

class ColorInfo {
public:
	vector<double> hsv[3]; //[hue,sat,val]
	double max;
	double min;
	double mean;
	double std;
};

void calculateStaticValues(std::vector<double> colorInfo, double minMaxMeanStd[4]) {

	minMaxMeanStd[0] = *std::min_element(colorInfo.begin(), colorInfo.end());
	minMaxMeanStd[1] = *std::max_element(colorInfo.begin(), colorInfo.end());
	minMaxMeanStd[2] = std::accumulate(colorInfo.begin(), std::end(colorInfo), 0.0) / std::size(colorInfo);
	double mean = minMaxMeanStd[2];
	minMaxMeanStd[3] = (std::inner_product(colorInfo.begin(), colorInfo.end(), colorInfo.begin(), 0.0) - mean * mean * colorInfo.size()) / colorInfo.size();

}


void extract80HsvFeatures() {

	for (int trainTestItr = 0; trainTestItr < 1; trainTestItr++) {

		bool isTrainData = true;
		if (trainTestItr == 0) {
			isTrainData = true;
		}
		else {
			isTrainData = false;
		}

		string rootDir = "";
		string toDir = "";
		if (isTrainData) {
			rootDir = "C:/MUSIC_RECOMMENDATION/src_data/shots_OMV200/";
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV200_csv_shot_80hsv/";
		}
		else {
			rootDir = "C:/MUSIC_RECOMMENDATION/shot_detection/output_first_frames_of_shots/recommendation_test/";
			toDir = "C:/MUSIC_RECOMMENDATION/src_data/recommendation_test_features/csv_shot_80hsv/";
		}

		vector<string> videoList = Dir::readIncludingFolder(rootDir);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<vector<double>> hsvFeatures;

				vector<string> frameList = Dir::readOutOfFolder(rootDir + videoList[videoItr]);


				// 各フレームの取得
				for (int frameItr = 0; frameItr < frameList.size(); frameItr++) {
					if (frameList[frameItr] != "." && frameList[frameItr] != "..") {

						string srcPath = rootDir + videoList[videoItr] + "/" + frameList[frameItr];
						cv::Mat uSrc = cv::imread(srcPath);

						// hsvに変換
						cv::Mat uHsv;
						cv::cvtColor(uSrc, uHsv, CV_BGR2HSV);

						// 各ピクセルのhsvの値を出力
						cv::Mat channels[3];
						cv::split(uHsv, channels);

						vector<double> blackWhite[2];
						ColorInfo colors[6];

						vector<double> tmphsvFeatures;

						for (int y = 0; y < uSrc.rows; y++) {
							for (int x = 0; x < uSrc.cols; x++) {

								double hue = 2.0 * (double)channels[0].ptr<unsigned char>(y)[x]; //hue:0~180 -> 0~360
								double sat = (double)channels[1].ptr<unsigned char>(y)[x];
								double val = (double)channels[2].ptr<unsigned char>(y)[x];

		
								// どこの色相に所属するかを決定
								// 所属する色相のビンに対して彩度と輝度を割り振る
								// 白黒は別に求める
								if (val < 30) { // 黒に見える良い感じの値
									blackWhite[0].push_back(val);
								}
								else if (val > 220) { // 白に見える良い感じの値
									blackWhite[1].push_back(val);
								}
								else {
									if (hue <= 30 || hue >= 330) {
										colors[0].hsv[0].push_back(hue);
										colors[0].hsv[1].push_back(sat);
										colors[0].hsv[2].push_back(val);
									}
									else if (hue > 30 && hue <= 90) {
										colors[1].hsv[0].push_back(hue);
										colors[1].hsv[1].push_back(sat);
										colors[1].hsv[2].push_back(val);
									}
									else if (hue > 90 && hue <= 150) {
										colors[2].hsv[0].push_back(hue);
										colors[2].hsv[1].push_back(sat);
										colors[2].hsv[2].push_back(val);
									}
									else if (hue > 150 && hue <= 210) {
										colors[3].hsv[0].push_back(hue);
										colors[3].hsv[1].push_back(sat);
										colors[3].hsv[2].push_back(val);
									}
									else if (hue > 210 && hue <= 270) {
										colors[4].hsv[0].push_back(hue);
										colors[4].hsv[1].push_back(sat);
										colors[4].hsv[2].push_back(val);
									}
									else if (hue > 270 && hue < 330) {
										colors[5].hsv[0].push_back(hue);
										colors[5].hsv[1].push_back(sat);
										colors[5].hsv[2].push_back(val);
									}

								}
							}
						}

						// 得られたデータを保存
						for (int bwItr = 0; bwItr < 2; bwItr++) {

							int size  = blackWhite[bwItr].size();

							if (size == 0) {
								for (int zeroItr = 0; zeroItr < 4; zeroItr++) {
									tmphsvFeatures.push_back(-1);
								}
							}
							else {
								double minMaxMeanStd[4] = {};
								calculateStaticValues(blackWhite[bwItr], minMaxMeanStd);

								for (int itr = 0; itr < 4; itr++) {
									tmphsvFeatures.push_back(minMaxMeanStd[itr]);
								}
							}
						}

						for (int colorItr = 0; colorItr < 6; colorItr++) {

							int size = colors[colorItr].hsv[0].size();

							for (int hsvItr = 0; hsvItr < 3; hsvItr++) {

								if (size == 0) {
									for (int zeroItr = 0; zeroItr < 4; zeroItr++) {
										tmphsvFeatures.push_back(-1);
									}
								}
								else {
									double minMaxMeanStd[4] = {};
									calculateStaticValues(colors[colorItr].hsv[hsvItr], minMaxMeanStd);

									for (int itr = 0; itr < 4; itr++) {
										tmphsvFeatures.push_back(minMaxMeanStd[itr]);
									}
								}
							}
						}

						hsvFeatures.push_back(tmphsvFeatures);
					}

				}


				cout << hsvFeatures.front().size() << endl;
				cout << hsvFeatures.size() << endl;

				// csvで出力
				string csv_name = toDir + videoList[videoItr] +"_shot_hsv80.csv";
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