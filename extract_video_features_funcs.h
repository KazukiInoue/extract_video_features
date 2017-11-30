#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Windows.h>

#include <opencv2/opencv.hpp>

using namespace std;

void extract20HsvFeatures();
void extract60ColorThemes();
void extract80HsvFeatures();
void extract768ColorHistogram(string colorSpace);
void extract4608HsvFeatures();
void kMeansColorSubtraction(cv::Mat &dst, std::vector<std::vector<double>> &clusterInfo, cv::Mat src, const int clusterNum);
void selectPrincipalColorThemes(vector<vector<double>> &colorThemes, cv::Mat src, const int clusterNum, const int numToExtract, vector<vector<double>> clusterInfo);
