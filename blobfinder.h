#ifndef BLOBFINDER_H
#define BLOBFINDER_H

#include <vector>
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>

class BlobFinder
{
private:
    float num_sigma_precalculated;
    float max_sigma_precalculated;
    float min_sigma_precalculated;
    std::vector<float> sigma_precalculated;
    std::vector<std::vector<float>> precalculated_kernels_gauss;
    std::vector<std::vector<float>> precalculated_kernels_laplacian;
    cv::Mat _get_peak_mask(cv::Mat image, int kernelSize, float threshold);
    std::vector<std::tuple<int,int,float>> _get_high_intensity_peaks(cv::Mat image);

public:
    BlobFinder();
    BlobFinder(float min_sigma, float max_sigma, int num_sigma);
    std::vector<std::tuple<int,int,float>> blob_log(cv::Mat image, float min_sigma, float max_sigma, int num_sigma, float threshold, bool exclude_border);
    cv::Mat LoGFilter(cv::Mat image, float sigma);
};

#endif // BLOBFINDER_H
