#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

using namespace std;


void extract60ColorThemes() {

	const int width = 256;
	const int height = 256;

	const int clusterNum = 20;
	const int numToExtract = 20;

	string rootDir[2] = {};
	string toDir[2] = {};

	// category=0:OMV200
	rootDir[0] = "../../src_data/shots_OMV200/";
	toDir[0] = "../../src_data/train_features/OMV200_csv_shot_60color_themes/";

	// category=1:recommendation_test
	rootDir[1] = "../../src_data/shots_recommendation_test/";
	toDir[1] = "../../src_data/recommendation_test_features/csv_shot_60color_themes/";

	for (int categoryItr = 0; categoryItr < 1; categoryItr++) {

		vector<string> videoList = Dir::readIncludingFolder(rootDir[categoryItr]);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> imgList = Dir::readOutOfFolder(rootDir[categoryItr] + videoList[videoItr]);

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

						kMeansColorSubtraction(/*&*/subtImg, /*&*/clusterInfo, uSrc, clusterNum);
						selectPrincipalColorThemes(/*&*/tmpColorThemes, subtImg, clusterNum, numToExtract, clusterInfo);

						colorThemes.push_back(tmpColorThemes);
					}
				}

				// csvで出力
				string csv_name = toDir[categoryItr] + videoList[videoItr] + "_shot_60color_themes.csv";
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
			}
		}
	}
}
