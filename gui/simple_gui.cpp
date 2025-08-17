#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

// Include algorithm headers
#include "advanced/filtering/guided_filter.hpp"
#include "advanced/filtering/side_window_filter.hpp"
#include "advanced/correction/gamma_correction.hpp"
#include "advanced/defogging/dark_channel.hpp"
#include "advanced/effects/cartoon_effect.hpp"
#include "advanced/effects/oil_painting_effect.hpp"

using namespace cv;
using namespace std;

// Global variables
Mat original_image, processed_image;
bool image_loaded = false;
string current_algorithm = "None";

// Algorithm parameters
struct AlgorithmParams {
    // Guided Filter
    float guided_radius = 60.0f;
    float guided_eps = 0.0001f;

    // Side Window Filter
    float side_window_radius = 5.0f;
    float side_window_sigma = 1.0f;

    // Gamma Correction
    float gamma_value = 1.0f;

    // Dark Channel Defogging
    int dark_channel_patch_size = 15;
    float dark_channel_omega = 0.95f;
    float dark_channel_t0 = 0.1f;

    // Cartoon Effect
    int cartoon_num_down = 2;
    int cartoon_num_bilateral = 7;

    // Oil Painting Effect
    int oil_painting_radius = 4;
    int oil_painting_intensity_levels = 20;
} params;

// Function to load image
void load_image(const string& path) {
    original_image = imread(path);
    if (!original_image.empty()) {
        image_loaded = true;
        processed_image = original_image.clone();
        cout << "Image loaded: " << path << endl;
    } else {
        cout << "Failed to load image: " << path << endl;
    }
}

// Function to apply selected algorithm
void apply_algorithm() {
    if (!image_loaded) return;

    processed_image = original_image.clone();

    if (current_algorithm == "Guided Filter") {
        ip101::advanced::guided_filter(original_image, processed_image,
                                     params.guided_radius, params.guided_eps);
    }
    else if (current_algorithm == "Side Window Filter") {
        ip101::advanced::side_window_filter(original_image, processed_image,
                                          params.side_window_radius, params.side_window_sigma);
    }
    else if (current_algorithm == "Gamma Correction") {
        ip101::advanced::standard_gamma_correction(original_image, processed_image,
                                                 params.gamma_value);
    }
    else if (current_algorithm == "Dark Channel Defogging") {
        ip101::advanced::dark_channel_defogging(original_image, processed_image,
                                              params.dark_channel_patch_size,
                                              params.dark_channel_omega,
                                              params.dark_channel_t0);
    }
    else if (current_algorithm == "Cartoon Effect") {
        ip101::advanced::cartoon_effect(original_image, processed_image,
                                      params.cartoon_num_down, params.cartoon_num_bilateral);
    }
    else if (current_algorithm == "Oil Painting Effect") {
        ip101::advanced::oil_painting_effect(original_image, processed_image,
                                           params.oil_painting_radius,
                                           params.oil_painting_intensity_levels);
    }
}

// Function to save processed image
void save_image() {
    if (!processed_image.empty()) {
        string filename = "output/simple_gui_processed_" + current_algorithm + ".jpg";
        filesystem::create_directories("output");
        imwrite(filename, processed_image);
        cout << "Image saved: " << filename << endl;
    }
}

// Function to create trackbar callbacks
void on_guided_radius(int value, void* userdata) {
    params.guided_radius = value;
    if (image_loaded && current_algorithm == "Guided Filter") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_guided_eps(int value, void* userdata) {
    params.guided_eps = value / 100000.0f;
    if (image_loaded && current_algorithm == "Guided Filter") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_side_window_radius(int value, void* userdata) {
    params.side_window_radius = value;
    if (image_loaded && current_algorithm == "Side Window Filter") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_side_window_sigma(int value, void* userdata) {
    params.side_window_sigma = value / 10.0f;
    if (image_loaded && current_algorithm == "Side Window Filter") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_gamma(int value, void* userdata) {
    params.gamma_value = value / 10.0f;
    if (image_loaded && current_algorithm == "Gamma Correction") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_dark_channel_patch_size(int value, void* userdata) {
    params.dark_channel_patch_size = value * 2 + 1; // Ensure odd number
    if (image_loaded && current_algorithm == "Dark Channel Defogging") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_dark_channel_omega(int value, void* userdata) {
    params.dark_channel_omega = value / 100.0f;
    if (image_loaded && current_algorithm == "Dark Channel Defogging") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_dark_channel_t0(int value, void* userdata) {
    params.dark_channel_t0 = value / 100.0f;
    if (image_loaded && current_algorithm == "Dark Channel Defogging") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_cartoon_down(int value, void* userdata) {
    params.cartoon_num_down = value;
    if (image_loaded && current_algorithm == "Cartoon Effect") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_cartoon_bilateral(int value, void* userdata) {
    params.cartoon_num_bilateral = value;
    if (image_loaded && current_algorithm == "Cartoon Effect") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_oil_painting_radius(int value, void* userdata) {
    params.oil_painting_radius = value;
    if (image_loaded && current_algorithm == "Oil Painting Effect") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

void on_oil_painting_levels(int value, void* userdata) {
    params.oil_painting_intensity_levels = value;
    if (image_loaded && current_algorithm == "Oil Painting Effect") {
        apply_algorithm();
        imshow("Processed Image", processed_image);
    }
}

// Function to setup trackbars for current algorithm
void setup_trackbars() {
    // Destroy existing trackbars
    destroyWindow("Parameters");

    // Create parameter window
    namedWindow("Parameters", WINDOW_AUTOSIZE);

    if (current_algorithm == "Guided Filter") {
        createTrackbar("Radius", "Parameters", nullptr, 100, on_guided_radius);
        setTrackbarPos("Radius", "Parameters", params.guided_radius);

        createTrackbar("Epsilon x100000", "Parameters", nullptr, 1000, on_guided_eps);
        setTrackbarPos("Epsilon x100000", "Parameters", params.guided_eps * 100000);
    }
    else if (current_algorithm == "Side Window Filter") {
        createTrackbar("Radius", "Parameters", nullptr, 20, on_side_window_radius);
        setTrackbarPos("Radius", "Parameters", params.side_window_radius);

        createTrackbar("Sigma x10", "Parameters", nullptr, 50, on_side_window_sigma);
        setTrackbarPos("Sigma x10", "Parameters", params.side_window_sigma * 10);
    }
    else if (current_algorithm == "Gamma Correction") {
        createTrackbar("Gamma x10", "Parameters", nullptr, 30, on_gamma);
        setTrackbarPos("Gamma x10", "Parameters", params.gamma_value * 10);
    }
    else if (current_algorithm == "Dark Channel Defogging") {
        createTrackbar("Patch Size", "Parameters", nullptr, 15, on_dark_channel_patch_size);
        setTrackbarPos("Patch Size", "Parameters", (params.dark_channel_patch_size - 1) / 2);

        createTrackbar("Omega x100", "Parameters", nullptr, 100, on_dark_channel_omega);
        setTrackbarPos("Omega x100", "Parameters", params.dark_channel_omega * 100);

        createTrackbar("T0 x100", "Parameters", nullptr, 50, on_dark_channel_t0);
        setTrackbarPos("T0 x100", "Parameters", params.dark_channel_t0 * 100);
    }
    else if (current_algorithm == "Cartoon Effect") {
        createTrackbar("Down Samples", "Parameters", nullptr, 5, on_cartoon_down);
        setTrackbarPos("Down Samples", "Parameters", params.cartoon_num_down);

        createTrackbar("Bilateral Iterations", "Parameters", nullptr, 15, on_cartoon_bilateral);
        setTrackbarPos("Bilateral Iterations", "Parameters", params.cartoon_num_bilateral);
    }
    else if (current_algorithm == "Oil Painting Effect") {
        createTrackbar("Radius", "Parameters", nullptr, 10, on_oil_painting_radius);
        setTrackbarPos("Radius", "Parameters", params.oil_painting_radius);

        createTrackbar("Intensity Levels", "Parameters", nullptr, 50, on_oil_painting_levels);
        setTrackbarPos("Intensity Levels", "Parameters", params.oil_painting_intensity_levels);
    }
}

int main() {
    cout << "=== IP101 Simple Image Processing GUI ===" << endl;
    cout << "Controls:" << endl;
    cout << "  '1-6': Select algorithm" << endl;
    cout << "  'l': Load image" << endl;
    cout << "  's': Save processed image" << endl;
    cout << "  'q': Quit" << endl;
    cout << endl;

    // Create windows
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Processed Image", WINDOW_AUTOSIZE);

    // Load default image
    load_image("assets/imori.jpg");

    if (image_loaded) {
        imshow("Original Image", original_image);
        imshow("Processed Image", processed_image);
    }

    char key;
    while (true) {
        key = waitKey(1) & 0xFF;

        if (key == 'q') {
            break;
        }
        else if (key == 'l') {
            // Load image dialog
            string filename = "assets/imori.jpg";
            load_image(filename);
            if (image_loaded) {
                imshow("Original Image", original_image);
                imshow("Processed Image", processed_image);
                apply_algorithm();
                setup_trackbars();
            }
        }
        else if (key == 's') {
            save_image();
        }
        else if (key == '1') {
            current_algorithm = "Guided Filter";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
        else if (key == '2') {
            current_algorithm = "Side Window Filter";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
        else if (key == '3') {
            current_algorithm = "Gamma Correction";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
        else if (key == '4') {
            current_algorithm = "Dark Channel Defogging";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
        else if (key == '5') {
            current_algorithm = "Cartoon Effect";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
        else if (key == '6') {
            current_algorithm = "Oil Painting Effect";
            cout << "Selected: " << current_algorithm << endl;
            apply_algorithm();
            imshow("Processed Image", processed_image);
            setup_trackbars();
        }
    }

    destroyAllWindows();
    return 0;
}
