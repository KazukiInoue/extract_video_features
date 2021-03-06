#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

using namespace std;


void extract768ColorHistogram(string colorSpace) {

	if (colorSpace != "bgr" &&colorSpace != "hsv"&&colorSpace != "lab") {

		std::cerr << colorSpace << " can't be dealed with by this program!" << endl;
		exit(1);
	}

	const int width = 256;
	const int height = 256;

	string rootDir[2] = { "../../src_data/shots_OMV200_improved/" ,
						  "../../src_data/shots_OMV62of65_improved/" };

	string toDir[2] = { "../../src_data/train_features/OMV200_csv_shot_768" + colorSpace + "/",
					    "../../src_data/train_features/OMV62of65_csv_shot_768" + colorSpace + "/" };

	for (int categoryItr = 0; categoryItr < 2; categoryItr++) {

		vector<string> videoList = Dir::readIncludingFolder(rootDir[categoryItr]);

		// videoXフォルダに入る
		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				vector<string> imgList = Dir::readExcludingFolder(rootDir[categoryItr] + videoList[videoItr]);

				vector<vector<int>> colorHist;

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

						cv::Mat uDst;
						if (colorSpace == "bgr") {
							uDst = uSrc.clone();
						}
						else if (colorSpace == "hsv") {
							cv::cvtColor(uSrc, uDst, CV_BGR2HSV_FULL);
						}
						else if (colorSpace == "lab") {
							cv::cvtColor(uSrc, uDst, CV_BGR2Lab);
						}

						//各画像の3つのチャンネルの値をそれぞれ256ビンのヒストグラムとして出力
						cv::Mat uChannels[3];
						cv::split(uDst, uChannels);

						vector<int> tmpColorHist(3 * 256, 0);

						for (int y = 0; y < uDst.rows; y++) {
							for (int x = 0; x < uDst.cols; x++) {

								int value0 = (int)uChannels[0].ptr<unsigned char>(y)[x];
								int value1 = (int)uChannels[1].ptr<unsigned char>(y)[x];
								int value2 = (int)uChannels[2].ptr<unsigned char>(y)[x];

								tmpColorHist[0 * 256 + value0] += 1;
								tmpColorHist[1 * 256 + value1] += 1;
								tmpColorHist[2 * 256 + value2] += 1;
							}
						}

						colorHist.push_back(tmpColorHist);
					}
				}

				// サイズの確認
				cout << colorHist.size() << endl;
				cout << colorHist.front().size() << endl;

				// csvで出力
				string csv_name = toDir[categoryItr] + videoList[videoItr] + "_shot_768" + colorSpace + ".csv";

				ofstream ofs(csv_name);
				for (int imgItr = 0; imgItr < colorHist.size(); imgItr++) {
					for (int valueItr = 0; valueItr < colorHist.front().size(); valueItr++) {

						if (valueItr == colorHist.front().size() - 1) {
							ofs << colorHist[imgItr][valueItr] << endl;
						}
						else {
							ofs << colorHist[imgItr][valueItr] << ",";
						}
					}
				}
			}
		}
	}
}