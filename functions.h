#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Windows.h>

#include <opencv2/opencv.hpp>

void extract20HsvFeatures();
void extract60ColorThemes();
void extract80HsvFeatures();
void extract768BgrFeatures();
void extract768HsvFeatures();
void extract768LabFeatures();
void extract768ColorHistogram(string colorSpace);
void extract4608HsvFeatures();