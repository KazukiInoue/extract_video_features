using namespace std;

#include "accessDirectory.h"
#include "functions.h"


void extract20HsvFeatures() {

	// ���[�g�t�H���_�̎w��
	//string rootDir = "C:/C++Projects/ResizeVideo/build/output_beat";
	string rootDir = "C:/C++Projects/ResizeVideo/build/output_recommendation_test";
	// string rootDir = "C:/Users/LAB/Desktop/test";

	vector<string> videoList = Dir::readIncludingFolder(rootDir);

	// videoX�t�H���_�ɓ���
	for (int videoItr = 0; videoItr < videoList.size(); videoItr++) {
		if (videoList[videoItr] != "." && videoList[videoItr] != "..") {

			vector<vector<double>> outData;

			vector<string> frameList = Dir::readOutOfFolder(rootDir + "/" + videoList[videoItr]);


			// �e�t���[���̎擾
			for (int frameItr = 0; frameItr < frameList.size(); frameItr++) {
				if (frameList[frameItr] != "." && frameList[frameItr] != "..") {

					string srcPath = rootDir + "/" + videoList[videoItr] + "/" + frameList[frameItr];
					cv::Mat src = cv::imread(srcPath);

					// hsv�ɕϊ�
					cv::Mat hsv;
					cv::cvtColor(src, hsv, CV_BGR2HSV);

					// �e�s�N�Z����hsv�̒l���o��
					cv::Mat channels[3];
					cv::split(hsv, channels);

					double eachHue[6][3] = {};
					double blackWhite[2] = {};
					vector<double> tmpOutData;

					for (int y = 0; y < src.rows; y++) {
						for (int x = 0; x < src.cols; x++) {

							double hue = 2.0 * (double)channels[0].ptr<unsigned char>(y)[x];
							double sat = (double)channels[1].ptr<unsigned char>(y)[x];
							double val = (double)channels[2].ptr<unsigned char>(y)[x];

							double blue = (double)src.ptr<cv::Vec3b>(y)[x][0];
							double green = (double)src.ptr<cv::Vec3b>(y)[x][1];
							double red = (double)src.ptr<cv::Vec3b>(y)[x][2];

							// �ǂ��̐F���ɏ������邩������
							// ��������F���̃r���ɑ΂��čʓx�ƋP�x������U��
							// �����͕ʂɋ��߂�
							if (blue < 40 && green < 40 && red < 40) {
								blackWhite[0] += 1;
							}
							else if (blue > 210 && green > 210 && red > 210) {
								blackWhite[1] += 1;
							}
							else {
								if (hue <= 30 || hue >= 330) {
									eachHue[0][0] += 1;
									eachHue[0][1] += sat;
									eachHue[0][2] += val;
								}
								else if (hue > 30 && hue <= 90) {
									eachHue[1][0] += 1;
									eachHue[1][1] += sat;
									eachHue[1][2] += val;
								}
								else if (hue > 90 && hue <= 150) {
									eachHue[2][0] += 1;
									eachHue[2][1] += sat;
									eachHue[2][2] += val;
								}
								else if (hue > 150 && hue <= 210) {
									eachHue[3][0] += 1;
									eachHue[3][1] += sat;
									eachHue[3][2] += val;
								}
								else if (hue > 210 && hue <= 270) {
									eachHue[4][0] += 1;
									eachHue[4][1] += sat;
									eachHue[4][2] += val;
								}
								else if (hue > 270 && hue < 330) {
									eachHue[5][0] += 1;
									eachHue[5][1] += sat;
									eachHue[5][2] += val;
								}

							}



						}
					}

					// ����ꂽ�f�[�^��ۑ�
					tmpOutData.push_back(blackWhite[0]);
					tmpOutData.push_back(blackWhite[1]);

					for (int c = 0; c < 6; c++) {
						double meanSat = eachHue[c][1] / eachHue[c][0];
						double meanVal = eachHue[c][2] / eachHue[c][0];

						if (eachHue[c][0] == 0) {
							meanSat = 0;
							meanVal = 0;
						}


						tmpOutData.push_back(eachHue[c][0]);
						tmpOutData.push_back(meanSat);
						tmpOutData.push_back(meanVal);
					}

					outData.push_back(tmpOutData);
				}

			}


			cout << outData.front().size() << endl;
			cout << outData.size() << endl;

			// csv�ŏo��
			string csv_name = "C:/C++Projects/get_video_features/build/output_video_features/20hsv_features/train/" + videoList[videoItr] + ".csv";
			ofstream ofs(csv_name);
			for (int dataItr = 0; dataItr < outData.size(); dataItr++) {
				for (int featureItr = 0; featureItr < outData.front().size(); featureItr++) {

					if (featureItr == outData.front().size() - 1) {
						ofs << outData[dataItr][featureItr] << endl;
					}
					else {
						ofs << outData[dataItr][featureItr] << ",";
					}

				}
			}
		}
	}
}