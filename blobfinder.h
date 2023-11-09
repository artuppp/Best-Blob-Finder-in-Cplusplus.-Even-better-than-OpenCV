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
    cv::Mat _get_peak_mask(cv::Mat image, int kernelSize, float threshold);
    std::vector<std::tuple<int,int,float>> _get_high_intensity_peaks(cv::Mat image);

public:
    BlobFinder() {};
    std::vector<std::tuple<int,int,float>> blob_log(cv::Mat image, float min_sigma, float max_sigma, int num_sigma, float threshold, bool exclude_border);
};

#endif // BLOBFINDER_H
