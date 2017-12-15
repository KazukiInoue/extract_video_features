#include "accessDirectory.h"
#include "extract_video_features_funcs.h"

using namespace std;


void extract512ColorHistogram(string colorSpace) {

	if (colorSpace != "bgr" &&colorSpace != "hsv"&&colorSpace != "lab") {

		std::cerr << colorSpace << " can't be dealed with by this program!" << endl;
		exit(1);
	}

	const int width = 256;
	const int height = 256;

	const int bins[3] = { 8, 8, 8 };

	string rootDir[2] = { "../../src_data/shots_OMV200_improved/",
						  "../../src_data/shots_OMV62of65_improved/" };

	string toDir[2] = { "../../src_data/train_features/OMV200_csv_shot_512" + colorSpace + "/",
						"../../src_data/train_features/OMV62of65_csv_shot_512" + colorSpace + "/" };

	for (int categoryItr = 1; categoryItr < 2; categoryItr++) {

		// videoXフォルダに入る
		vector<string> videoList = Dir::readIncludingFolder(rootDir[categoryItr]);

		for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
			if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

				// 各画像の取得
				vector<string> imgList = Dir::readExcludingFolder(rootDir[categoryItr] + videoList[videoItr]);

				vector<vector<int>> colorHist;


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

						vector<int> tmpColorHist(bins[0] * bins[1] * bins[2], 0);

						for (int y = 0; y < uDst.rows; y++) {
							for (int x = 0; x < uDst.cols; x++) {

								int value[3] = { (int)uChannels[0].ptr<unsigned char>(y)[x],
												 (int)uChannels[1].ptr<unsigned char>(y)[x],
												 (int)uChannels[2].ptr<unsigned char>(y)[x] };

								// intで値を丸め込むことで、どのビンに所属するのかを決定
								int belong[3];
								for (int c = 0; c < 3; c++) {
									belong[c] = value[c] / (256 / bins[c]);
									if (belong[c] == bins[c]) { // 0 <= belong[c] <= bins[c]-1
										belong[c] = bins[c] - 1;
									}
								}

								int index = belong[0] + bins[0] * belong[1] + bins[0] * bins[1] * belong[2];

								tmpColorHist[index] += 1;
							}
						}

						colorHist.push_back(tmpColorHist);
					}
				}

				// サイズの確認
				cout << colorHist.size() << endl;
				cout << colorHist.front().size() << endl;

				// csvで出力
				string csv_name = toDir[categoryItr] + videoList[videoItr] + "_shot_512" + colorSpace + ".csv";

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