#ifndef IMAGE_INPAINTING_HPP
#define IMAGE_INPAINTING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Diffusion-based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param radius Diffusion radius
 * @param num_iterations Number of iterations
 * @return Inpainted image
 */
cv::Mat diffusion_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int radius = 3,
    int num_iterations = 100);

/**
 * @brief Patch-matching based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param patch_size Patch size
 * @param search_area Search area size
 * @return Inpainted image
 */
cv::Mat patch_match_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 9,
    int search_area = 30);

/**
 * @brief Fast marching method based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param radius Inpainting radius
 * @return Inpainted image
 */
cv::Mat fast_marching_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int radius = 3);

/**
 * @brief Texture synthesis based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param patch_size Texture patch size
 * @param overlap Overlap region size
 * @return Inpainted image
 */
cv::Mat texture_synthesis_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 15,
    int overlap = 4);

/**
 * @brief Structure propagation based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param patch_size Patch size
 * @param num_iterations Number of iterations
 * @return Inpainted image
 */
cv::Mat structure_propagation_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 9,
    int num_iterations = 10);

/**
 * @brief PatchMatch based image inpainting
 * @param src Input image
 * @param mask Inpainting mask (255 indicates regions to be inpainted)
 * @param patch_size Patch size
 * @param num_iterations Number of iterations
 * @return Inpainted image
 */
cv::Mat patchmatch_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 7,
    int num_iterations = 5);

/**
 * @brief Video inpainting
 * @param frames Input video frames
 * @param masks Inpainting masks for each frame
 * @param patch_size Patch size
 * @param num_iterations Number of iterations
 * @return Inpainted video frames
 */
std::vector<cv::Mat> video_inpaint(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& masks,
    int patch_size = 7,
    int num_iterations = 5);

} // namespace ip101

#endif // IMAGE_INPAINTING_HPP