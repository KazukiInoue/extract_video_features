// In emAlgorighmColorSubtraction.cpp, I referred to http://seiya-kumada.blogspot.jp/2013/03/em-opencv.html

# include <numeric>

#include "extract_video_features_funcs.h"

constexpr double Epsilon = 1.0e-08;

using namespace std;
using namespace cv;
using namespace cv::ml;


void observeProbs(const Mat& probs)
{
	vector<double> t(probs.cols, 0.0);
	for (int n = 0; n < probs.rows; ++n) {
		const double* gamma_n = probs.ptr<double>(n);
		double s = 0.0;
		for (int k = 0; k < probs.cols; ++k) {
			s += gamma_n[k]; // 
			t[k] += gamma_n[k];
		}
		assert(std::abs(s - 1.0) < Epsilon);
	}
	double total = std::accumulate(t.begin(), t.end(), 0.0);
	assert(std::abs(total - probs.rows) < Epsilon);
}

void observeWeights(const Mat& weights)
{
	MatConstIterator_<double> first = weights.begin<double>();
	MatConstIterator_<double> last = weights.end<double>();
	double s = 0.0;
	while (first != last) { // loop over k
		s += *first; // *first means \pi_{k}
		++first;
	}
	assert(std::abs(s - 1.0) < Epsilon);
}

void observeLabelsAndMeans(Mat& subtImg, std::vector<std::vector<double>> &clusterInfo, const Mat& means, const Mat labels, int height, int width, const int clusterNum)
{
	const int dimension = 3;

	subtImg = Mat(Size(height, width), CV_8UC3);

	Mat means_u8;
	means.convertTo(means_u8, CV_8UC1, 255.0);
	Mat means_u8c3 = means_u8.reshape(dimension);

	vector<double> pixelNum(clusterNum, (2, 0));
	vector<vector<double>> posiCenter(clusterNum, vector<double>(2, 0));

	for (int y = 0; y < subtImg.rows; y++) {
		for (int x = 0; x < subtImg.cols; x++) {


			// 減色した画像を生成
			int label = labels.ptr<int>(x + y*subtImg.cols)[0];
			subtImg.ptr<Vec3b>(y)[x] = means_u8c3.ptr<Vec3b>(label)[0];

			// 後で使うクラスタの位置情報をここで収集
			pixelNum[label] += 1.0;
			posiCenter[label][0] += x;
			posiCenter[label][1] += y;
		}
	}

	// クラスタリングの情報を収集
	for (int clusterItr = 0; clusterItr < clusterNum; clusterItr++) {

		double xCenter = posiCenter[clusterItr][0] / pixelNum[clusterItr];
		double yCenter = posiCenter[clusterItr][1] / pixelNum[clusterItr];
		double pixelRatio = pixelNum[clusterItr] / (subtImg.cols * subtImg.rows);

		double blueCenter = (double)means_u8c3.ptr<Vec3b>(clusterItr)[0][0];
		double greenCenter = (double)means_u8c3.ptr<Vec3b>(clusterItr)[0][1];
		double redCenter = (double)means_u8c3.ptr<Vec3b>(clusterItr)[0][2];

		vector<double> tmpClusterInfo;
		tmpClusterInfo.push_back(xCenter);
		tmpClusterInfo.push_back(yCenter);
		tmpClusterInfo.push_back(pixelRatio);
		tmpClusterInfo.push_back(blueCenter);
		tmpClusterInfo.push_back(greenCenter);
		tmpClusterInfo.push_back(redCenter);
		clusterInfo.push_back(tmpClusterInfo);
	}
}

void emAlgorithmColorSubtraction(Mat& subtImg, std::vector<std::vector<double>> &clusterInfo, Mat src, const int clusterNum) {

	assert(src.type() == CV_8UC3);
	const int image_rows = src.rows;
	const int image_cols = src.cols;

	constexpr int dimension = 3;

	Mat reshaped_image = src.reshape(1, image_rows * image_cols);
	assert(reshaped_image.type() == CV_8UC1);
	assert(reshaped_image.rows == image_rows * image_cols);
	assert(reshaped_image.cols == dimension);

	// create an input for the EM Algorithm
	Mat samples;
	reshaped_image.convertTo(samples, CV_64FC1, 1.0 / 255.0);
	assert(samples.type() == CV_64FC1);
	assert(samples.rows == image_rows * image_cols);
	assert(samples.cols == dimension);

	Ptr<EM> model = EM::create();
	model->setClustersNumber(clusterNum);

	// prepare outputs
	Mat labels;
	Mat probs;

	model->trainEM(samples, noArray(), labels, probs);

	assert(labels.type() == CV_32SC1);
	assert(labels.rows == image_rows * image_cols);
	assert(labels.cols == 1);

	assert(probs.type() == CV_64FC1);
	assert(probs.rows == image_rows * image_cols);
	assert(probs.cols == clusterNum);
	// observeProbs(/*&*/probs);

	const Mat means = model->getMeans();

	assert(means.type() == CV_64FC1);
	assert(means.rows == clusterNum);
	assert(means.cols == dimension);
	observeLabelsAndMeans(/*&*/subtImg,/*&*/clusterInfo,/*&*/means, labels, image_rows, image_cols, clusterNum);

	const Mat weights = model->getWeights();

	assert(weights.type() == CV_64FC1);
	assert(weights.rows == 1);
	assert(weights.cols == clusterNum);
	// observeWeights(/*&*/weights);

}