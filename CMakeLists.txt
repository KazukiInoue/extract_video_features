cmake_minimum_required(VERSION 2.8)

#プロジェクト名の指定
project(extract_video_features)

#Debug,Releaseの指定,MinSizeRelとRelWithDebInfoの選択肢を生成しない
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

#OpenCVのディレクトリの指定
set(OpenCV_DIR "C:/opencv-3.3.0/build")
include_directories("C:/opencv-3.3.0/build/include")

#パッケージの検索
find_package(OpenCV REQUIRED)

#インクルードパスの指定
include_directories( ${OpenCV_INCLUDE_DIRS})
add_executable(extract_video_features main.cpp extract20HsvFeatures.cpp extract60ColorThemes.cpp extract80HsvFeatures.cpp extract768BgrFeatures.cpp extract768HsvFeatures.cpp extract768ColorHistogram.cpp extract768LabFeatures.cpp extract4608HsvFeatures.cpp)
#リリース版の設定
set(CMAKE_BUILD_TYPE Debug;Release)

#リンクするライブラリの追加
target_link_libraries(extract_video_features ${OpenCV_LIBS})