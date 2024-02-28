# BlobFinder

Example with downscaling

´´´
 cv::Mat imageRawWB2;
   cv::resize(imageRawWB, imageRawWB2, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
//    Inicializate image to 3 dimensional zeros
    auto reflexes = blobFinder.blob_log(imageRawWB2, 5, 20, 4, 0.1, false);
    Logger::Log("WavefrontAnalyzer::processHS", "end blob finder");
//    Reescale reflexes to original size
    std::transform(reflexes.begin(), reflexes.end(), reflexes.begin(), [](std::tuple<int, int, float>& x) {
        // Correct radius (= gaussian standard deviation) with sqrt(2):
        std::get<0>(x) = std::get<0>(x) * 4;
        std::get<1>(x) = std::get<1>(x) * 4;
        std::get<2>(x) = std::get<2>(x) * 4;
        return x;
    });
´´´

HEADER:

´´´
std::vector<std::tuple<int,int,float>> blob_log(cv::Mat image, float min_sigma, float max_sigma, int num_sigma, float threshold, bool exclude_border);
´´´


More information in: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log
  
