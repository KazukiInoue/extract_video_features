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
add_executable(extract_video_features 
em_algorithm_color_subtraction.cpp
extract_20hsv_features.cpp
extract_60color_themes.cpp
extract_80hsv_features.cpp
extract_512color_histogram.cpp
extract_768color_histogram.cpp
extract_4608hsv_features.cpp
k-means_color_subtraction.cpp
main.cpp
select_principal_color_themes.cpp)

#リリース版の設定
set(CMAKE_BUILD_TYPE Debug;Release)

#リンクするライブラリの追加
target_link_libraries(extract_video_features ${OpenCV_LIBS})