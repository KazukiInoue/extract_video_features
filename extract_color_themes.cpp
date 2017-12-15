#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

using namespace std;

void bgr2OtherColorSpace(vector<vector<vector<double>>> &colorThemes, string colorSpace) {

	for (int imgItr = 0; imgItr < colorThemes.size(); imgItr++) {
		for (int themeItr = 0; themeItr < colorThemes.front().size(); themeItr++) {

			vector<double> bgr = colorThemes[imgItr][themeItr];

			cv::Mat dst(cv::Size(1, 1), CV_8UC3);
			cv::Mat channels[3];
			cv::split(dst, channels);

			channels[0] = (uchar)bgr[0];
			channels[1] = (uchar)bgr[1];
			channels[2] = (uchar)bgr[2];

			cv::merge(channels, 3, dst);
			if (colorSpace == "hsv") {
				cv::cvtColor(dst, dst, CV_BGR2HSV);
			}
			else if (colorSpace == "lab") {
				cv::cvtColor(dst, dst, CV_BGR2Lab);
			}


			for (int c = 0; c < 3; c++) {
				colorThemes[imgItr][themeItr][c] = (double)dst.ptr<cv::Vec3b>(0)[0][c];
			}
		}
	}
}


void extractColorThemes(string colorSpace, string mlType) {

	if (mlType != "kMeans" && mlType != "em") {
		cerr << "We haven't dealed with " << mlType << " algorithm type yet." << endl;
		exit(1);
	}

	if (colorSpace != "bgr" && colorSpace != "hsv" && colorSpace != "lab") {
		std::cerr << colorSpace << " can't be dealed with by thie program!" << endl;
		exit(1);
	}

	const int width = 256;
	const int height = 256;

	const int clusterNum = 5;
	const int numToExtract = 5;

	string rootDir[2] = {};
	string toDir[2] = {};

	// category=0:OMV200
	rootDir[0] = "../../src_data/shots_OMV200_improved/";
	toDir[0] = "../../src_data/train_features/OMV200_csv_shot_60color_themes_" + colorSpace + "/";

	// category=1:recommendation_test
	rootDir[1] = "../../src_data/shots_OMV62of65_improved/";
	toDir[1] = "../../src_data/train_features/csv_shot_60color_themes_" + colorSpace + "/";

	for (int categoryItr = 0; categoryItr < 2; categoryItr++) {

		vector<string> videoList = Dir::readIncludingFolder(rootDir[categoryItr]);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> imgList = Dir::readExcludingFolder(rootDir[categoryItr] + videoList[videoItr]);

				vector<vector<vector<double>>> colorThemes;

				// 各画像の取得
				for (int imgItr = 0; imgItr < imgList.size(); imgItr++) {
					if (imgList[imgItr] != "." && imgList[imgItr] != "..") {

						string srcPath = rootDir[categoryItr] + videoList[videoItr] + "/" + imgList[imgItr];

						cv::Mat uSrc = cv::imread(srcPath);
						if (uSrc.empty()) {
							std::cerr << "uSrc doesn't exist!" << endl;
							exit(1);
						}

						cv::resize(uSrc, uSrc, cv::Size(), width / (double)uSrc.cols, height / (double)uSrc.rows);

						cv::Mat subtImg(uSrc.size(), uSrc.type());
						vector<vector<double>> clusterInfo; /*select_principal_color_themesで使用する*/
						vector<vector<double>> tmpColorThemes(numToExtract, vector<double>(3, 0));

						if (mlType == "kMeans") {
							kMeansColorSubtraction(/*&*/subtImg, /*&*/clusterInfo, uSrc, clusterNum);
						}
						else if (mlType == "em") {
							emAlgorithmColorSubtraction(/*&*/subtImg, /*&*/clusterInfo, uSrc, clusterNum);
						}

						selectPrincipalColorThemes(/*&*/tmpColorThemes, subtImg, clusterNum, numToExtract, clusterInfo);

						colorThemes.push_back(tmpColorThemes);
					}
				}

				// BGR以外の色空間に変換

				if (colorSpace == "bgr") {}
				else if (colorSpace == "hsv") {
					bgr2OtherColorSpace(colorThemes, colorSpace);
				}
				else if (colorSpace == "lab") {
					bgr2OtherColorSpace(colorThemes, colorSpace);
				}
				else {
					std::cerr << colorSpace << " can't be dealed with by thie program!" << endl;
					exit(1);
				}


				// csvで出力
				string csv_name = toDir[categoryItr] + "/" + videoList[videoItr] + "_shot_" + to_string(clusterNum * 3) + "color_themes_" + colorSpace + ".csv";
				ofstream ofs(csv_name);

				int imgNum = colorThemes.size();

				for (int imgItr = 0; imgItr < imgNum; imgItr++) {
					for (int themeItr = 0; themeItr < numToExtract; themeItr++) {
						for (int c = 0; c < 3; c++) {
							if (themeItr == numToExtract - 1 && c == 2) {
								ofs << colorThemes[imgItr][themeItr][c] << endl;
							}
							else {
								ofs << colorThemes[imgItr][themeItr][c] << ",";
							}
						}
					}
				}

				cout << "終了" << endl;
			}
		}
	}
}
