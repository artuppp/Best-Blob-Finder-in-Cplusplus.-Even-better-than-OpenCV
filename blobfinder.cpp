#include "blobfinder.h"
#include "execution"


/**
 * @brief Code for calculate the gaussian kernel in 1D
 * @param sigma
 * @param order
 * @param radius
 * @return
 */
std::vector<float> gaussian_kernel1d(float sigma, int order, int radius) {
    if (order < 0) {
        throw std::invalid_argument("order must be non-negative");
    }

    std::vector<float> kernel(2 * radius + 1);
    std::vector<float> exponent_range(order + 1);
    float sigma2 = sigma * sigma;

    std::iota(exponent_range.begin(), exponent_range.end(), 0);

    //    for (int i = -radius; i <= radius; ++i) {
    //        float x = static_cast<float>(i);
    //        float phi_x = std::exp(-0.5 / (sigma2) * (x * x));
    //        kernel[i + radius] = phi_x;
    //    }
    std::iota(kernel.begin(), kernel.end(), -radius);
    std::transform(std::execution::par_unseq, kernel.begin(), kernel.end(), kernel.begin(), [sigma2](float& x) {
        return std::exp(-0.5 / (sigma2) * (x * x));
    });

    float sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    std::transform(std::execution::par_unseq, kernel.begin(), kernel.end(), kernel.begin(), [sum](float& x) { return x / sum; });

    if (order == 0) {
        return kernel;
    } else {
        // Calculate q(x) * phi(x) using matrix operators
        std::vector<float> q(order + 1, 0.0);
        std::vector<float> q_deriv(order + 1, 0.0);
        q[0] = 1.0;

        std::vector<float> D((order + 1) * (order + 1));
        std::vector<float> P((order + 1) * (order + 1));
        // D = np.diag(exponent_range[1:], 1)
        for (int i = 0; i < order; ++i) {
            int index = i * (order + 1) + i + 1;
            if (index < D.size()) {
                D[index] = exponent_range[i + 1];
            }
        }
        // P = np.diag(np.ones(order)/-sigma2,-1)
        for (int i = 0; i < order; ++i) {
            int index = (i + 1) * (order + 1) + i;
            if (index < D.size()) {
                P[index] = -1.0 / sigma2;
            }
        }
        // D = D + P
        for (int i = 0; i < (order + 1) * (order + 1); ++i) {
            D[i] += P[i];
        }
        for (int i = 0; i < order; i++){
            // q = Q_deriv.dot(q)
            for (int j = 0; j < order + 1; j++){
                q_deriv[j] = std::inner_product(D.begin() + j * (order + 1), D.begin() + (j + 1) * (order + 1), q.begin(), 0.0);
            }
            q = q_deriv;
        }
        // a = (x[:, None] ** exponent_range)
        std::vector<float> a((2 * radius + 1) * (order + 1), 0.0);
        for (int i = 0; i < 2 * radius + 1; ++i) {
            for (int j = 0; j < order + 1; ++j) {
                a[i * (order + 1) + j] = std::pow(i - radius, j);
            }
        }
        // q = a.dot(q)
        std::vector<float> newQ(2 * radius + 1, 0.0);
        for (int i = 0; i < 2 * radius + 1; ++i) {
            newQ[i] = std::inner_product(a.begin() + i * (order + 1), a.begin() + (i + 1) * (order + 1), q.begin(), 0.0);
        }
        // q = q * phi_x
        for (int i = 0; i < 2 * radius + 1; ++i) {
            newQ[i] *= kernel[i];
        }
        return newQ;
    }
}

/**
 * @brief BlobFinder::BlobFinder default constructor
 */
BlobFinder::BlobFinder()
{
    this->num_sigma_precalculated = 0;
    this->max_sigma_precalculated = 0;
    this->min_sigma_precalculated = 0;
}

/**
 * @brief BlobFinder::BlobFinder constructor
 * @param min_sigma
 * @param max_sigma
 * @param num_sigma
 */
BlobFinder::BlobFinder(float min_sigma, float max_sigma, int num_sigma)
{
    this->num_sigma_precalculated = num_sigma;
    this->max_sigma_precalculated = max_sigma;
    this->min_sigma_precalculated = min_sigma;
    // Precalculate sigma values
    float sigma_step = (max_sigma - min_sigma) / (num_sigma - 1);
    for (int i = 0; i < num_sigma; i++)
    {
        this->sigma_precalculated.push_back(min_sigma + i * sigma_step);
    }
    for (int i = 0; i < num_sigma; i++)
    {
        int ksize = int(4 * sigma_precalculated[i] + 0.5);
        this->precalculated_kernels_gauss.push_back(gaussian_kernel1d(sigma_precalculated[i], 0, ksize));
        this->precalculated_kernels_laplacian.push_back(gaussian_kernel1d(sigma_precalculated[i], 2, ksize));
    }
}

/**
 * @brief dilate1D Maximum filter in 1D
 * @param src
 * @param dst
 * @param length
 * @param stride
 * @param width
 */
inline void dilate1D (float *src, float *dst, int length, int stride, int width)
{
    std::transform(std::execution::par_unseq, src + width * stride, src - width * stride + length, dst + width * stride, [width, stride](float& x) {
        float max = 0;
        for (int j = -width; j <= width; j++)
        {
            max = std::max(max, (&x)[j * stride]);
        }
        return max;
    });
}

/**
 * @brief Maximum filter
 * @param image (n-dimensional)
 * @param width
 * @return
 */
cv::Mat dilateNd (cv::Mat image, int width)
{
    cv::Mat dst = image.clone();
    cv::Mat dst_aux = image.clone();
    float *dst_ptr = (float *)dst.data;
    float *dst_aux_ptr = (float *)dst_aux.data;
    std::fill(std::execution::par_unseq, dst_aux_ptr, dst_aux_ptr + image.total(), 0.0);
    for (int i = 0; i < image.dims; i++)
    {
        dilate1D(dst_ptr, dst_aux_ptr, image.total(), image.step[i] / sizeof(float), width);
        // Swap pointers
        std::copy(std::execution::par_unseq, dst_aux_ptr, dst_aux_ptr + image.total(), dst_ptr);
        std::fill(std::execution::par_unseq, dst_aux_ptr, dst_aux_ptr + image.total(), 0.0);
    }
    return dst;
}

/**
 * @brief Return the highest intensity peak coordinates.
 * @param image
 * @param min_distance
 * @return
 */
std::vector<std::tuple<int,int,float>> BlobFinder::_get_high_intensity_peaks(cv::Mat image)
{
    // Tuple with 3d point is defined as: z,y,x
    std::vector<std::tuple<int,int,float>> peaks;
    // Get coordinates of peaks
    for (int d = 0; d < image.size[0]; d++)
    {
        for (int i = 0; i < image.size[1]; i++)
        {
            for (int j = 0; j < image.size[2]; j++)
            {
                auto point = std::make_tuple(d, i, j);
                int index = i * image.step[1] / sizeof(float) + j * image.step[2] / sizeof(float) + d * image.step[0] / sizeof(float);
                float value = ((float *)image.data)[index];
                if (value != 0)
                {
                    peaks.push_back(point);
                }
            }
        }
    }
    return peaks;
}

/**
 * @brief Return the mask containing all peak candidates above thresholds.
 * @param image
 * @param kernelSize The radius of the kernel
 * @param threshold
 * @param mask
 * @return
 */
cv::Mat BlobFinder::_get_peak_mask(cv::Mat image, int kernelSize, float threshold)
{
    // Maximum filter
    cv::Mat max_image = dilateNd(image, kernelSize);
    // Print images
    cv::Mat out = image.clone();
    // out = image == image_max;
    std::transform(std::execution::par_unseq, image.begin<float>(), image.end<float>(), max_image.begin<float>(), out.begin<float>(), [](float& x, float& y) {
        return (x == y)? 1.0 : 0.0;
    });
    // if np.all(out): out[:] = false
    if (std::all_of(std::execution::par_unseq, out.begin<float>(), out.end<float>(), [](float i) { return i == 1;}))
    {
        std::fill(std::execution::par_unseq, out.begin<float>(), out.end<float>(), 0.0);
    }
    // out = out & (image >= threshold)
    std::transform(std::execution::par_unseq, image.begin<float>(), image.end<float>(), out.begin<float>(), out.begin<float>() , [threshold](float& x, float& y) {
        return (x >= threshold) * y;
    });
    return out;
}


/**
 * @brief Kernel for computing the Convolution1D
 * @param image
 * @param kernel
 * @param axis
 * @return
 */
cv::Mat Convolution1D(cv::Mat image, std::vector<float> kernel, int axis)
{
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    int ksize = kernel.size();
    int radius = ksize / 2;
    int stride = image.step[axis] / sizeof(float);
    std::transform(std::execution::par_unseq, image.begin<float>() + stride * radius, image.end<float>() - stride * radius,
                   result.begin<float>() + stride * radius, [kernel, radius, stride](float& x) {
                       const float * kernel_ptr = kernel.data();
                       kernel_ptr += radius;
                       float sum = (float)(&x)[0] * kernel_ptr[0];
                       for (int k = 1; k < radius; k++)
                       {
                           float izq = (float)(&x)[k * stride];
                           float der = (float)(&x)[-k * stride];
                           sum += (izq + der) * kernel_ptr[k];
                       }
                       return (float)sum;
                   });
    return result;
}

/**
 * @brief Kernel for computing the LoGFilter
 * @param image
 * @param sigma
 * @return
 */
cv::Mat BlobFinder::LoGFilter(cv::Mat image, float sigma)
{
    // Check if sigma is in the precalculated list
    auto it = std::find(sigma_precalculated.begin(), sigma_precalculated.end(), sigma);
    if (it != sigma_precalculated.end())
    {
        int index = std::distance(sigma_precalculated.begin(), it);
        return Convolution1D(Convolution1D(image, precalculated_kernels_gauss[index], 0), precalculated_kernels_laplacian[index], 1);
    }
    int ksize = int(4 * sigma + 0.5);
    std::vector<float> deriv2kernel = gaussian_kernel1d(sigma, 2, ksize);
    std::vector<float> kernel = gaussian_kernel1d(sigma, 0, ksize);
    cv::Mat result = Convolution1D(image, deriv2kernel, 0);
    result = Convolution1D(result, kernel, 1);
    return result;
}

std::vector<std::tuple<int, int, float>> BlobFinder::blob_log(cv::Mat image, float min_sigma, float max_sigma, int num_sigma, float threshold, bool exclude_border)
{
    // Pass image to float
    image.convertTo(image, CV_32F);
    float maxValue = *std::max_element(std::execution::par_unseq, image.begin<float>(), image.end<float>());
    std::transform(std::execution::par_unseq, image.begin<float>(), image.end<float>(), image.begin<float>(), [maxValue](float& x) {
        return x / maxValue;
    });
    // Create sigma list from min to max with num_sigma elements
    std::vector<float> sigma_list;
    float sigma_step = (max_sigma - min_sigma) / (num_sigma - 1);
    for (int i = 0; i < num_sigma; i++)
    {
        sigma_list.push_back(min_sigma + i * sigma_step);
    }
    // Compute gaussian laplace with different sigmas
    int dims[] = {num_sigma + 2, image.rows, image.cols};
    cv::Mat cube(3, dims, CV_32F, cv::Scalar::all(0));
    // First slice is padding
    cv::Mat emptySlice = cv::Mat::zeros(image.size(), CV_32F);
    std::copy(std::execution::par_unseq, emptySlice.begin<float>(), emptySlice.end<float>(), cube.begin<float>());
    for (int i = 1; i < num_sigma; i++)
    {
        //ndi.gaussian_laplace(image, s)
        cv::Mat slice = LoGFilter(image, sigma_list[i-1]);
        //-ndi.gaussian_laplace(image, s) * np.mean(s)**2
        int currentSigma = sigma_list[i];
        std::transform(std::execution::par_unseq, slice.begin<float>(), slice.end<float>(), slice.begin<float>(), [currentSigma](float& x) {
            return -x * currentSigma * currentSigma;
        });
        // Copy slice to cube
        std::copy(std::execution::par_unseq, slice.begin<float>(), slice.end<float>(), cube.begin<float>() + i * image.rows * image.cols);
    }
    // Last slice is padding
    std::copy(std::execution::par_unseq, emptySlice.begin<float>(), emptySlice.end<float>(), cube.begin<float>() + (num_sigma + 1) * emptySlice.rows * emptySlice.cols);
    // Compute local maximas
    // Compute peak mask
    int size = 1;
    cv::Mat peak_mask = _get_peak_mask(cube, size, threshold);
    // Get peaks (sigma, y, x)
    auto peaks = _get_high_intensity_peaks(peak_mask);
    // Compute blob radius
    if (peaks.empty())
    {
        return peaks;
    }
    // Translate sigma and replace
    // sigmas_of_peaks = sigma_list[local_maxima[:, -1]]
    // lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
    std::transform(std::execution::par_unseq, peaks.begin(), peaks.end(), peaks.begin(), [sigma_list](std::tuple<int, int, float>& x) {
        // Correct radius (= gaussian standard deviation) with sqrt(2):
        std::get<0>(x) = sigma_list[std::get<0>(x) - 1] * std::sqrt(2);
        return x;
    });
    return peaks;
}
