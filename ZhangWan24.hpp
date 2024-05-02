/*
* Author of the code: Mária Nováková
* Author of the paper: Zhang L., Wan Y.
* Date of implementation: 28.04.2024
* Description: Implementation of color-to-gray image conversion using salient colors and radial basis functions.
*/

#ifndef ZHANGWAN24_HPP
#define ZHANGWAN24_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace std;
using namespace cv;


class ColorToGrayConverter {
public:
    void quantizeColors(cv::Mat &image, int &k, int max_k, float theta_0, float theta_1);
    void ordering(cv::Mat image);
    void createGrayScale(cv::Mat image, float sigma);

private:
    cv::Vec3f computePrincipalDirection(const cv::Mat& image);
    void expandCentroids(cv::Vec3f c_0, int &k, cv::Mat img, std::vector<cv::Vec3f> *centers);
    float euclideanDistance(const cv::Vec3f& color1, const cv::Vec3f& color2);
    std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusterImage(cv::Mat image, std::vector<cv::Vec3f> centroids);
    void actualizeCenters(std::vector<cv::Vec3f> *centers, std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusters);
    void MSE(std::vector<cv::Vec3f> centers, std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusters, std::vector<float> *individualMSE);
    float MSE_k(cv::Mat image, std::vector<cv::Vec3f> centers, std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusters);
    float g(Mat image, vector<pair<Point, Vec3f>> cluster);
    float MSEG_k(cv::Mat image, std::vector<cv::Vec3f> centers, std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusters);
    float Entropy(cv::Mat img);
    bool classification(float E);
    float M_k(float MSE_k, float MSEG_k);
    float weightedEuclidean(const cv::Vec3f& color1, const cv::Vec3f& color2);
    float gaussianKernel(const cv::Vec3f& color1, const cv::Vec3f& color2, float sigma);
    float laplaceKernel(const cv::Vec3f& color1, const cv::Vec3f& color2, float sigma);
    float clamp(float value);
    float getGreyValue(cv::Vec3f img_color, std::vector<float> a, float sigma);
    cv::Mat convertToGrayQuantizedImage(const cv::Mat& originalImage, const std::vector<float>& centroids, const std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>>& clusters);
    cv::Mat convertToQuantizedImage(const cv::Mat& originalImage, const std::vector<cv::Vec3f>& centroids, const vector<vector<pair<Point, Vec3f>>>& clusters);

    // parameters
    std::vector<cv::Vec3f> centers; 
    std::vector<std::vector<std::pair<cv::Point, cv::Vec3f>>> clusters;
    std::vector<float> grayvalues;
};

#endif
