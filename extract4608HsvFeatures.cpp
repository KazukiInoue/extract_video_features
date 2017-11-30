#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

using namespace std;

void extract4608HsvFeatures() {

	bool isTrainData = true;
	string rootDir = "";
	string toDir = "";
	if (isTrainData) {
		rootDir = "C:/MUSIC_RECOMMENDATION/src_data/shots_OMV200/";
		toDir = "C:/MUSIC_RECOMMENDATION/src_data/train_features/OMV200_csv_4608hsv/";
	}
	else {
		rootDir = "../../shot_detection/output_first_frames_of_shots/recommendation_test/";
		toDir = "../output_video_features/shot/recommendation_test/4608hsv_features/";
	}

	vector<string> videoList = Dir::readIncludingFolder(rootDir);

	// ヒストグラムのbinの数を指定

	const int bin[3] = { 18,  16, 16 };

	// videoXフォルダに入る
	for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
		if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

			vector<string> imgList = Dir::readOutOfFolder(rootDir + videoList[videoItr]);

			vector<vector<int>> hsvFeatures;

			// 各フレームの取得
			for (int imgItr = 0; imgItr < imgList.size(); imgItr++) {
				if (imgList[imgItr] != "." && imgList[imgItr] != "..") {

					string srcPath = rootDir + videoList[videoItr] + "/" + imgList[imgItr];

					cv::Mat uBgr = cv::imread(srcPath);
					cv::resize(uBgr, uBgr, cv::Size(), 256 / (double)uBgr.cols, 256 / (double)uBgr.rows);
					cv::Mat uHsv;
					cv::cvtColor(uBgr, uHsv, CV_BGR2HSV);


					// 各ピクセルのhsvの値を出力
					cv::Mat uChannels[3];
					cv::split(uHsv, uChannels);

					int hue = 0, sat = 0, val = 0;

					vector<vector<vector<int>>> dHsv3dHist(bin[0], vector<vector<int>>(bin[1], vector<int>(bin[2], 0)));
					vector<int> tmpHsvFeatures;

					//各フレームの色相・彩度・輝度を抽出
					for (int y = 0; y < uHsv.rows; y++) {
						for (int x = 0; x < uHsv.cols; x++) {

							hue = (int)uChannels[0].ptr<unsigned char>(y)[x];  // hue:0~180
							sat = (int)uChannels[1].ptr<unsigned char>(y)[x];
							val = (int)uChannels[2].ptr<unsigned char>(y)[x];

							int index[3] = { hue / (180 / bin[0]), sat / (256 / bin[1]), val / (256 / bin[2]) };

							// ヒストグラムの値を更新
							if (index[0] == bin[0]) {
								index[0] = bin[0] - 1;
							}
							if (index[1] == bin[1]) {
								index[1] = bin[1] - 1;
							}
							if (index[2] == bin[2]) {
								index[2] = bin[2] - 1;
							}

							dHsv3dHist[index[0]][index[1]][index[2]] += 1;
						}
					}

					//データを格納
					for (int chIter = 0; chIter < dHsv3dHist.size(); chIter++) {
						for (int y = 0; y < dHsv3dHist.front().size(); y++) {
							for (int x = 0; x < dHsv3dHist.front().front().size(); x++) {

								tmpHsvFeatures.push_back(dHsv3dHist[chIter][y][x]);

							}
						}
					}

					hsvFeatures.push_back(tmpHsvFeatures);
				}
			}


			cout << hsvFeatures.front().size() << endl;
			cout << hsvFeatures.size() << endl;

			// csvで出力
			string csv_name = toDir + videoList[videoItr] + "_shot_hsv4608.csv";

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