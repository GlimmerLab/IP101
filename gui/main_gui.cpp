#include <opencv2/opencv.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

// Include all algorithm headers
#include "advanced/filtering/guided_filter.hpp"
#include "advanced/filtering/side_window_filter.hpp"
#include "advanced/filtering/homomorphic_filter.hpp"
#include "advanced/correction/automatic_white_balance.hpp"
#include "advanced/correction/gamma_correction.hpp"
#include "advanced/defogging/dark_channel.hpp"
#include "advanced/defogging/fast_defogging.hpp"
#include "advanced/enhancement/hdr.hpp"
#include "advanced/effects/cartoon_effect.hpp"
#include "advanced/effects/oil_painting_effect.hpp"
#include "advanced/detection/color_cast_detection.hpp"

using namespace cv;
using namespace std;

// Global variables
GLFWwindow* window = nullptr;
Mat original_image, processed_image;
bool image_loaded = false;
string current_algorithm = "None";
bool show_original = true;
bool show_processed = true;

// UI State
bool show_toolbar = true;
bool show_algorithm_panel = true;
bool show_parameters_panel = true;
bool show_history_panel = false;
bool show_info_panel = false;
bool show_layers_panel = false;
float zoom_level = 1.0f;
ImVec2 pan_offset(0, 0);
bool is_panning = false;
ImVec2 last_mouse_pos;

// History for undo/redo
struct HistoryEntry {
    Mat image;
    string algorithm;
    string timestamp;
};
vector<HistoryEntry> history;
int current_history_index = -1;

// Algorithm parameters
struct AlgorithmParams {
    // Guided Filter
    float guided_radius = 60.0f;
    float guided_eps = 0.0001f;

    // Side Window Filter
    float side_window_radius = 5.0f;
    float side_window_sigma = 1.0f;

    // Homomorphic Filter
    float homomorphic_gamma_low = 0.3f;
    float homomorphic_gamma_high = 1.5f;
    float homomorphic_cutoff = 30.0f;

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

    // Color Cast Detection
    float color_cast_threshold = 0.15f;
} params;

// Function to convert OpenCV Mat to ImGui texture
void* mat_to_texture(const Mat& mat) {
    if (mat.empty()) return nullptr;

    Mat display_mat;
    if (mat.channels() == 3) {
        cvtColor(mat, display_mat, COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cvtColor(mat, display_mat, COLOR_GRAY2RGB);
    } else {
        display_mat = mat.clone();
    }

    // Create OpenGL texture
    GLuint texture_id;
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, 0x812F); // GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, 0x812F); // GL_CLAMP_TO_EDGE

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, display_mat.cols, display_mat.rows,
                 0, GL_RGB, GL_UNSIGNED_BYTE, display_mat.data);

    return (void*)(intptr_t)texture_id;
}

// Function to load image
void load_image(const string& path) {
    original_image = imread(path);
    if (!original_image.empty()) {
        image_loaded = true;
        processed_image = original_image.clone();

        // Reset view
        zoom_level = 1.0f;
        pan_offset = ImVec2(0, 0);

        // Clear history and add initial state
        history.clear();
        HistoryEntry entry;
        entry.image = original_image.clone();
        entry.algorithm = "Original";
        entry.timestamp = "Loaded";
        history.push_back(entry);
        current_history_index = 0;

        cout << "Image loaded: " << path << endl;
    } else {
        cout << "Failed to load image: " << path << endl;
    }
}

// Function to add to history
void add_to_history(const Mat& image, const string& algorithm) {
    // Remove any entries after current index
    if (current_history_index < history.size() - 1) {
        history.erase(history.begin() + current_history_index + 1, history.end());
    }

    HistoryEntry entry;
    entry.image = image.clone();
    entry.algorithm = algorithm;
    entry.timestamp = "Applied";
    history.push_back(entry);
    current_history_index = history.size() - 1;

    // Limit history size
    if (history.size() > 20) {
        history.erase(history.begin());
        current_history_index--;
    }
}

// Function to apply selected algorithm
void apply_algorithm() {
    if (!image_loaded) return;

    Mat previous_image = processed_image.clone();
    processed_image = original_image.clone();

    if (current_algorithm == "Guided Filter") {
        ip101::advanced::guided_filter(original_image, original_image, processed_image,
                                     static_cast<int>(params.guided_radius), params.guided_eps);
    }
    else if (current_algorithm == "Side Window Filter") {
        ip101::advanced::side_window_filter(original_image, processed_image,
                                          static_cast<int>(params.side_window_radius),
                                          ip101::advanced::SideWindowType::BOX);
    }
    else if (current_algorithm == "Homomorphic Filter") {
        ip101::advanced::guided_filter(original_image, original_image, processed_image,
                                     static_cast<int>(params.homomorphic_cutoff), params.homomorphic_gamma_low);
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
        ip101::advanced::CartoonParams cartoon_params;
        cartoon_params.edge_size = params.cartoon_num_down;
        cartoon_params.median_blur_size = params.cartoon_num_bilateral;
        ip101::advanced::cartoon_effect(original_image, processed_image, cartoon_params);
    }
    else if (current_algorithm == "Oil Painting Effect") {
        ip101::advanced::OilPaintingParams oil_params;
        oil_params.radius = params.oil_painting_radius;
        oil_params.levels = params.oil_painting_intensity_levels;
        ip101::advanced::oil_painting_effect(original_image, processed_image, oil_params);
    }
    else if (current_algorithm == "Color Cast Detection") {
        ip101::advanced::ColorCastResult result;
        ip101::advanced::ColorCastDetectionParams detect_params;
        detect_params.threshold = params.color_cast_threshold;
        ip101::advanced::detect_color_cast(original_image, result, detect_params);

        if (result.has_color_cast) {
            Mat correction_matrix = Mat::eye(3, 3, CV_32F);
            for (int i = 0; i < 3; i++) {
                correction_matrix.at<float>(i, i) = 1.0f - result.color_cast_vector[i];
            }
            transform(original_image, processed_image, correction_matrix);
        }
    }

    // Add to history if image changed
    if (!cv::countNonZero(processed_image != previous_image)) {
        add_to_history(processed_image, current_algorithm);
    }
}

// Function to save processed image
void save_image() {
    if (!processed_image.empty()) {
        string filename = "output/gui_processed_" + current_algorithm + ".jpg";
        filesystem::create_directories("output");
        imwrite(filename, processed_image);
        cout << "Image saved: " << filename << endl;
    }
}

// Professional Photoshop-like GUI style
void setup_professional_style() {
    ImGuiStyle& style = ImGui::GetStyle();

    // Colors - Professional dark theme
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_Border]                 = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.00f, 0.60f, 0.90f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_Separator]              = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.00f, 0.60f, 0.90f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.00f, 0.70f, 1.00f, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(0.00f, 0.70f, 1.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

    // Style
    style.WindowPadding = ImVec2(8, 8);
    style.FramePadding = ImVec2(6, 4);
    style.CellPadding = ImVec2(6, 6);
    style.ItemSpacing = ImVec2(8, 6);
    style.ScrollbarSize = 14;
    style.GrabMinSize = 10;
    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = 0;
    style.TabBorderSize = 0;
    style.WindowRounding = 6;
    style.ChildRounding = 4;
    style.FrameRounding = 4;
    style.PopupRounding = 4;
    style.ScrollbarRounding = 9;
    style.GrabRounding = 4;
    style.TabRounding = 4;
}

// Render toolbar with professional icons
void render_toolbar() {
    if (!show_toolbar) return;

    ImGui::Begin("Toolbar", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoTitleBar);
    float toolbar_height = 50;
    ImGui::SetWindowSize(ImVec2(ImGui::GetWindowWidth(), toolbar_height));

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 0));

    // File operations
    if (ImGui::Button("📁 Open", ImVec2(70, 30))) {
        load_image("assets/imori.jpg");
    }
    ImGui::SameLine();
    if (ImGui::Button("💾 Save", ImVec2(70, 30))) {
        save_image();
    }
    ImGui::SameLine();

    ImGui::Separator();
    ImGui::SameLine();

    // Edit operations
    if (ImGui::Button("↶ Undo", ImVec2(70, 30)) && current_history_index > 0) {
        current_history_index--;
        processed_image = history[current_history_index].image.clone();
    }
    ImGui::SameLine();
    if (ImGui::Button("↷ Redo", ImVec2(70, 30)) && current_history_index < history.size() - 1) {
        current_history_index++;
        processed_image = history[current_history_index].image.clone();
    }
    ImGui::SameLine();

    ImGui::Separator();
    ImGui::SameLine();

    // View operations
    if (ImGui::Button("🔍 100%", ImVec2(70, 30))) {
        zoom_level = 1.0f;
        pan_offset = ImVec2(0, 0);
    }
    ImGui::SameLine();
    if (ImGui::Button("📐 Fit", ImVec2(70, 30))) {
        // Fit to window
        zoom_level = 1.0f;
        pan_offset = ImVec2(0, 0);
    }
    ImGui::SameLine();

    ImGui::Separator();
    ImGui::SameLine();

    // Algorithm operations
    if (ImGui::Button("🔄 Reset", ImVec2(70, 30))) {
        if (image_loaded) {
            processed_image = original_image.clone();
            current_algorithm = "None";
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("✨ Apply", ImVec2(70, 30))) {
        if (image_loaded && current_algorithm != "None") {
            apply_algorithm();
        }
    }

    ImGui::PopStyleVar();
    ImGui::End();
}

// Render algorithm panel with categories
void render_algorithm_panel() {
    if (!show_algorithm_panel) return;

    ImGui::Begin("Algorithms", nullptr, ImGuiWindowFlags_None);

    if (ImGui::BeginTabBar("AlgorithmTabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
        if (ImGui::BeginTabItem("🎨 Filters")) {
            ImGui::BeginChild("FiltersChild", ImVec2(0, 0), true);

            if (ImGui::Selectable("🔍 Guided Filter", current_algorithm == "Guided Filter")) {
                current_algorithm = "Guided Filter";
                if (image_loaded) apply_algorithm();
            }
            if (ImGui::Selectable("🪟 Side Window Filter", current_algorithm == "Side Window Filter")) {
                current_algorithm = "Side Window Filter";
                if (image_loaded) apply_algorithm();
            }
            if (ImGui::Selectable("🌊 Homomorphic Filter", current_algorithm == "Homomorphic Filter")) {
                current_algorithm = "Homomorphic Filter";
                if (image_loaded) apply_algorithm();
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("⚙️ Correction")) {
            ImGui::BeginChild("CorrectionChild", ImVec2(0, 0), true);

            if (ImGui::Selectable("📊 Gamma Correction", current_algorithm == "Gamma Correction")) {
                current_algorithm = "Gamma Correction";
                if (image_loaded) apply_algorithm();
            }
            if (ImGui::Selectable("🎨 Color Cast Detection", current_algorithm == "Color Cast Detection")) {
                current_algorithm = "Color Cast Detection";
                if (image_loaded) apply_algorithm();
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("✨ Effects")) {
            ImGui::BeginChild("EffectsChild", ImVec2(0, 0), true);

            if (ImGui::Selectable("🎭 Cartoon Effect", current_algorithm == "Cartoon Effect")) {
                current_algorithm = "Cartoon Effect";
                if (image_loaded) apply_algorithm();
            }
            if (ImGui::Selectable("🖼️ Oil Painting Effect", current_algorithm == "Oil Painting Effect")) {
                current_algorithm = "Oil Painting Effect";
                if (image_loaded) apply_algorithm();
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("🌫️ Defogging")) {
            ImGui::BeginChild("DefoggingChild", ImVec2(0, 0), true);

            if (ImGui::Selectable("🌅 Dark Channel Defogging", current_algorithm == "Dark Channel Defogging")) {
                current_algorithm = "Dark Channel Defogging";
                if (image_loaded) apply_algorithm();
            }

            ImGui::EndChild();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }
    ImGui::End();
}

// Render parameters panel with better layout
void render_parameters_panel() {
    if (!show_parameters_panel) return;

    ImGui::Begin("Parameters", nullptr, ImGuiWindowFlags_None);

    if (current_algorithm != "None") {
        ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Current Algorithm: %s", current_algorithm.c_str());
        ImGui::Separator();

        ImGui::BeginChild("ParamsChild", ImVec2(0, 0), true);

        if (current_algorithm == "Guided Filter") {
            ImGui::Text("🔍 Guided Filter Parameters");
            ImGui::Spacing();

            ImGui::Text("Radius");
            if (ImGui::SliderFloat("##radius", &params.guided_radius, 1.0f, 100.0f, "%.1f")) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Epsilon");
            if (ImGui::SliderFloat("##epsilon", &params.guided_eps, 0.00001f, 0.01f, "%.5f")) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Side Window Filter") {
            ImGui::Text("🪟 Side Window Filter Parameters");
            ImGui::Spacing();

            ImGui::Text("Radius");
            if (ImGui::SliderFloat("##sw_radius", &params.side_window_radius, 1.0f, 20.0f, "%.1f")) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Sigma");
            if (ImGui::SliderFloat("##sw_sigma", &params.side_window_sigma, 0.1f, 5.0f, "%.1f")) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Homomorphic Filter") {
            ImGui::Text("🌊 Homomorphic Filter Parameters");
            ImGui::Spacing();

            ImGui::Text("Gamma Low");
            if (ImGui::SliderFloat("##gamma_low", &params.homomorphic_gamma_low, 0.1f, 1.0f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Gamma High");
            if (ImGui::SliderFloat("##gamma_high", &params.homomorphic_gamma_high, 1.0f, 3.0f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Cutoff");
            if (ImGui::SliderFloat("##cutoff", &params.homomorphic_cutoff, 10.0f, 100.0f, "%.1f")) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Gamma Correction") {
            ImGui::Text("📊 Gamma Correction Parameters");
            ImGui::Spacing();

            ImGui::Text("Gamma Value");
            if (ImGui::SliderFloat("##gamma", &params.gamma_value, 0.1f, 3.0f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Dark Channel Defogging") {
            ImGui::Text("🌅 Dark Channel Defogging Parameters");
            ImGui::Spacing();

            ImGui::Text("Patch Size");
            if (ImGui::SliderInt("##patch_size", &params.dark_channel_patch_size, 5, 31)) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Omega");
            if (ImGui::SliderFloat("##omega", &params.dark_channel_omega, 0.5f, 1.0f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("T0");
            if (ImGui::SliderFloat("##t0", &params.dark_channel_t0, 0.01f, 0.5f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Cartoon Effect") {
            ImGui::Text("🎭 Cartoon Effect Parameters");
            ImGui::Spacing();

            ImGui::Text("Down Samples");
            if (ImGui::SliderInt("##down_samples", &params.cartoon_num_down, 1, 5)) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Bilateral Iterations");
            if (ImGui::SliderInt("##bilateral_iter", &params.cartoon_num_bilateral, 1, 15)) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Oil Painting Effect") {
            ImGui::Text("🖼️ Oil Painting Effect Parameters");
            ImGui::Spacing();

            ImGui::Text("Radius");
            if (ImGui::SliderInt("##oil_radius", &params.oil_painting_radius, 1, 10)) {
                if (image_loaded) apply_algorithm();
            }

            ImGui::Text("Intensity Levels");
            if (ImGui::SliderInt("##intensity_levels", &params.oil_painting_intensity_levels, 5, 50)) {
                if (image_loaded) apply_algorithm();
            }
        }
        else if (current_algorithm == "Color Cast Detection") {
            ImGui::Text("🎨 Color Cast Detection Parameters");
            ImGui::Spacing();

            ImGui::Text("Threshold");
            if (ImGui::SliderFloat("##threshold", &params.color_cast_threshold, 0.05f, 0.5f, "%.2f")) {
                if (image_loaded) apply_algorithm();
            }
        }

        ImGui::EndChild();
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Select an algorithm to see parameters");
    }
    ImGui::End();
}

// Render image display with zoom and pan
void render_image_display() {
    ImGui::Begin("Image Display", nullptr, ImGuiWindowFlags_None);

    if (image_loaded) {
        // Handle mouse input for zoom and pan
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool is_hovered = ImGui::IsWindowHovered();

        if (is_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (!is_panning) {
                is_panning = true;
                last_mouse_pos = mouse_pos;
            } else {
                ImVec2 delta = ImVec2(mouse_pos.x - last_mouse_pos.x, mouse_pos.y - last_mouse_pos.y);
                pan_offset.x += delta.x;
                pan_offset.y += delta.y;
                last_mouse_pos = mouse_pos;
            }
        } else {
            is_panning = false;
        }

        // Mouse wheel zoom
        float wheel = ImGui::GetIO().MouseWheel;
        if (is_hovered && wheel != 0) {
            zoom_level += wheel * 0.1f;
            zoom_level = max(0.1f, min(zoom_level, 5.0f));
        }

        // Calculate display area
        ImVec2 window_size = ImGui::GetWindowSize();
        ImVec2 content_size = ImVec2(window_size.x - 20, window_size.y - 80);

        // Calculate image display size with zoom
        float image_aspect = (float)original_image.cols / original_image.rows;
        float window_aspect = content_size.x / content_size.y;

        ImVec2 base_display_size;
        if (image_aspect > window_aspect) {
            base_display_size.x = content_size.x;
            base_display_size.y = content_size.x / image_aspect;
        } else {
            base_display_size.y = content_size.y;
            base_display_size.x = content_size.y * image_aspect;
        }

        ImVec2 display_size(base_display_size.x * zoom_level, base_display_size.y * zoom_level);

        // Center the image
        ImVec2 image_pos = ImGui::GetCursorScreenPos();
        image_pos.x += (content_size.x - display_size.x) * 0.5f + pan_offset.x;
        image_pos.y += (content_size.y - display_size.y) * 0.5f + pan_offset.y;

        // Display images side by side
        ImGui::BeginChild("ImageContainer", content_size, true);

        if (show_original) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Original Image");
            void* original_texture = mat_to_texture(original_image);
            ImGui::SetCursorScreenPos(image_pos);
            ImGui::Image(original_texture, display_size);

            if (show_processed && !processed_image.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Processed Image");
                void* processed_texture = mat_to_texture(processed_image);
                ImGui::SetCursorScreenPos(ImVec2(image_pos.x + display_size.x + 20, image_pos.y));
                ImGui::Image(processed_texture, display_size);
            }
        } else if (show_processed && !processed_image.empty()) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Processed Image");
            void* processed_texture = mat_to_texture(processed_image);
            ImGui::SetCursorScreenPos(image_pos);
            ImGui::Image(processed_texture, display_size);
        }

        ImGui::EndChild();

        // Zoom info
        ImGui::Text("Zoom: %.1f%% | Pan: (%.0f, %.0f)", zoom_level * 100, pan_offset.x, pan_offset.y);
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No image loaded. Use File -> Open Image to start.");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Or click the 'Open' button in the toolbar.");
    }
    ImGui::End();
}

// Render history panel
void render_history_panel() {
    if (!show_history_panel) return;

    ImGui::Begin("History", nullptr, ImGuiWindowFlags_None);

    if (history.empty()) {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No history available");
    } else {
        for (int i = 0; i < history.size(); i++) {
            bool is_selected = (i == current_history_index);
            if (ImGui::Selectable(history[i].algorithm.c_str(), is_selected)) {
                current_history_index = i;
                processed_image = history[i].image.clone();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Click to restore this state");
            }
        }
    }
    ImGui::End();
}

// Render info panel
void render_info_panel() {
    if (!show_info_panel) return;

    ImGui::Begin("Info", nullptr, ImGuiWindowFlags_None);

    if (image_loaded) {
        ImGui::Text("Image Information:");
        ImGui::Separator();
        ImGui::Text("Dimensions: %dx%d", original_image.cols, original_image.rows);
        ImGui::Text("Channels: %d", original_image.channels());
        ImGui::Text("Type: %s", original_image.type() == CV_8UC3 ? "BGR" : "Grayscale");
        ImGui::Text("Memory: %.2f MB", (original_image.total() * original_image.elemSize()) / (1024.0f * 1024.0f));

        ImGui::Spacing();
        ImGui::Text("Current Algorithm: %s", current_algorithm.c_str());
        ImGui::Text("Zoom Level: %.1f%%", zoom_level * 100);

        if (!history.empty()) {
            ImGui::Spacing();
            ImGui::Text("History: %d/%d", current_history_index + 1, history.size());
        }
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No image loaded");
    }
    ImGui::End();
}

// Main GUI function with professional layout
void render_gui() {
    // Calculate layout dimensions
    ImVec2 display_size = ImGui::GetIO().DisplaySize;
    float toolbar_height = show_toolbar ? 60.0f : 0.0f;
    float status_height = 25.0f;
    float left_panel_width = show_algorithm_panel ? 250.0f : 0.0f;
    float right_panel_width = show_parameters_panel ? 300.0f : 0.0f;
    float bottom_panel_height = (show_history_panel || show_info_panel) ? 150.0f : 0.0f;

    float main_area_y = ImGui::GetFrameHeight() + toolbar_height;
    float main_area_height = display_size.y - main_area_y - status_height - bottom_panel_height;

    // Menu Bar
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Image", "Ctrl+O")) {
                load_image("assets/imori.jpg");
            }
            if (ImGui::MenuItem("Save Image", "Ctrl+S", false, !processed_image.empty())) {
                save_image();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                glfwSetWindowShouldClose(window, true);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, current_history_index > 0)) {
                if (current_history_index > 0) {
                    current_history_index--;
                    processed_image = history[current_history_index].image.clone();
                }
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, current_history_index < history.size() - 1)) {
                if (current_history_index < history.size() - 1) {
                    current_history_index++;
                    processed_image = history[current_history_index].image.clone();
                }
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Reset", "Ctrl+R", false, image_loaded)) {
                if (image_loaded) {
                    processed_image = original_image.clone();
                    current_algorithm = "None";
                }
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Toolbar", nullptr, &show_toolbar);
            ImGui::MenuItem("Algorithm Panel", nullptr, &show_algorithm_panel);
            ImGui::MenuItem("Parameters Panel", nullptr, &show_parameters_panel);
            ImGui::MenuItem("History Panel", nullptr, &show_history_panel);
            ImGui::MenuItem("Info Panel", nullptr, &show_info_panel);
            ImGui::Separator();
            ImGui::MenuItem("Original Image", nullptr, &show_original);
            ImGui::MenuItem("Processed Image", nullptr, &show_processed);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) {
                // Show about dialog
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Toolbar - Fixed at top
    if (show_toolbar) {
        ImGui::SetNextWindowPos(ImVec2(0, ImGui::GetFrameHeight()));
        ImGui::SetNextWindowSize(ImVec2(display_size.x, toolbar_height));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 5));
        ImGui::Begin("Toolbar", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 0));

        // File operations
        if (ImGui::Button("📁 Open", ImVec2(80, 35))) {
            load_image("assets/imori.jpg");
        }
        ImGui::SameLine();
        if (ImGui::Button("💾 Save", ImVec2(80, 35))) {
            save_image();
        }
        ImGui::SameLine();

        ImGui::Separator();
        ImGui::SameLine();

        // Edit operations
        if (ImGui::Button("↶ Undo", ImVec2(80, 35)) && current_history_index > 0) {
            current_history_index--;
            processed_image = history[current_history_index].image.clone();
        }
        ImGui::SameLine();
        if (ImGui::Button("↷ Redo", ImVec2(80, 35)) && current_history_index < history.size() - 1) {
            current_history_index++;
            processed_image = history[current_history_index].image.clone();
        }
        ImGui::SameLine();

        ImGui::Separator();
        ImGui::SameLine();

        // View operations
        if (ImGui::Button("🔍 100%", ImVec2(80, 35))) {
            zoom_level = 1.0f;
            pan_offset = ImVec2(0, 0);
        }
        ImGui::SameLine();
        if (ImGui::Button("📐 Fit", ImVec2(80, 35))) {
            zoom_level = 1.0f;
            pan_offset = ImVec2(0, 0);
        }
        ImGui::SameLine();

        ImGui::Separator();
        ImGui::SameLine();

        // Algorithm operations
        if (ImGui::Button("🔄 Reset", ImVec2(80, 35))) {
            if (image_loaded) {
                processed_image = original_image.clone();
                current_algorithm = "None";
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("✨ Apply", ImVec2(80, 35))) {
            if (image_loaded && current_algorithm != "None") {
                apply_algorithm();
            }
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
        ImGui::End();
    }

    // Left Panel - Algorithms
    if (show_algorithm_panel) {
        ImGui::SetNextWindowPos(ImVec2(0, main_area_y));
        ImGui::SetNextWindowSize(ImVec2(left_panel_width, main_area_height));
        ImGui::Begin("Algorithms", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (ImGui::BeginTabBar("AlgorithmTabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
            if (ImGui::BeginTabItem("🎨 Filters")) {
                ImGui::BeginChild("FiltersChild", ImVec2(0, 0), true);

                if (ImGui::Selectable("🔍 Guided Filter", current_algorithm == "Guided Filter")) {
                    current_algorithm = "Guided Filter";
                    if (image_loaded) apply_algorithm();
                }
                if (ImGui::Selectable("🪟 Side Window Filter", current_algorithm == "Side Window Filter")) {
                    current_algorithm = "Side Window Filter";
                    if (image_loaded) apply_algorithm();
                }
                if (ImGui::Selectable("🌊 Homomorphic Filter", current_algorithm == "Homomorphic Filter")) {
                    current_algorithm = "Homomorphic Filter";
                    if (image_loaded) apply_algorithm();
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("⚙️ Correction")) {
                ImGui::BeginChild("CorrectionChild", ImVec2(0, 0), true);

                if (ImGui::Selectable("📊 Gamma Correction", current_algorithm == "Gamma Correction")) {
                    current_algorithm = "Gamma Correction";
                    if (image_loaded) apply_algorithm();
                }
                if (ImGui::Selectable("🎨 Color Cast Detection", current_algorithm == "Color Cast Detection")) {
                    current_algorithm = "Color Cast Detection";
                    if (image_loaded) apply_algorithm();
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("✨ Effects")) {
                ImGui::BeginChild("EffectsChild", ImVec2(0, 0), true);

                if (ImGui::Selectable("🎭 Cartoon Effect", current_algorithm == "Cartoon Effect")) {
                    current_algorithm = "Cartoon Effect";
                    if (image_loaded) apply_algorithm();
                }
                if (ImGui::Selectable("🖼️ Oil Painting Effect", current_algorithm == "Oil Painting Effect")) {
                    current_algorithm = "Oil Painting Effect";
                    if (image_loaded) apply_algorithm();
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            if (ImGui::BeginTabItem("🌫️ Defogging")) {
                ImGui::BeginChild("DefoggingChild", ImVec2(0, 0), true);

                if (ImGui::Selectable("🌅 Dark Channel Defogging", current_algorithm == "Dark Channel Defogging")) {
                    current_algorithm = "Dark Channel Defogging";
                    if (image_loaded) apply_algorithm();
                }

                ImGui::EndChild();
                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
        ImGui::End();
    }

    // Right Panel - Parameters
    if (show_parameters_panel) {
        ImGui::SetNextWindowPos(ImVec2(display_size.x - right_panel_width, main_area_y));
        ImGui::SetNextWindowSize(ImVec2(right_panel_width, main_area_height));
        ImGui::Begin("Parameters", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (current_algorithm != "None") {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Current Algorithm: %s", current_algorithm.c_str());
            ImGui::Separator();

            ImGui::BeginChild("ParamsChild", ImVec2(0, 0), true);

            if (current_algorithm == "Guided Filter") {
                ImGui::Text("🔍 Guided Filter Parameters");
                ImGui::Spacing();

                ImGui::Text("Radius");
                if (ImGui::SliderFloat("##radius", &params.guided_radius, 1.0f, 100.0f, "%.1f")) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Epsilon");
                if (ImGui::SliderFloat("##epsilon", &params.guided_eps, 0.00001f, 0.01f, "%.5f")) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Side Window Filter") {
                ImGui::Text("🪟 Side Window Filter Parameters");
                ImGui::Spacing();

                ImGui::Text("Radius");
                if (ImGui::SliderFloat("##sw_radius", &params.side_window_radius, 1.0f, 20.0f, "%.1f")) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Sigma");
                if (ImGui::SliderFloat("##sw_sigma", &params.side_window_sigma, 0.1f, 5.0f, "%.1f")) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Homomorphic Filter") {
                ImGui::Text("🌊 Homomorphic Filter Parameters");
                ImGui::Spacing();

                ImGui::Text("Gamma Low");
                if (ImGui::SliderFloat("##gamma_low", &params.homomorphic_gamma_low, 0.1f, 1.0f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Gamma High");
                if (ImGui::SliderFloat("##gamma_high", &params.homomorphic_gamma_high, 1.0f, 3.0f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Cutoff");
                if (ImGui::SliderFloat("##cutoff", &params.homomorphic_cutoff, 10.0f, 100.0f, "%.1f")) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Gamma Correction") {
                ImGui::Text("📊 Gamma Correction Parameters");
                ImGui::Spacing();

                ImGui::Text("Gamma Value");
                if (ImGui::SliderFloat("##gamma", &params.gamma_value, 0.1f, 3.0f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Dark Channel Defogging") {
                ImGui::Text("🌅 Dark Channel Defogging Parameters");
                ImGui::Spacing();

                ImGui::Text("Patch Size");
                if (ImGui::SliderInt("##patch_size", &params.dark_channel_patch_size, 5, 31)) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Omega");
                if (ImGui::SliderFloat("##omega", &params.dark_channel_omega, 0.5f, 1.0f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("T0");
                if (ImGui::SliderFloat("##t0", &params.dark_channel_t0, 0.01f, 0.5f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Cartoon Effect") {
                ImGui::Text("🎭 Cartoon Effect Parameters");
                ImGui::Spacing();

                ImGui::Text("Down Samples");
                if (ImGui::SliderInt("##down_samples", &params.cartoon_num_down, 1, 5)) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Bilateral Iterations");
                if (ImGui::SliderInt("##bilateral_iter", &params.cartoon_num_bilateral, 1, 15)) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Oil Painting Effect") {
                ImGui::Text("🖼️ Oil Painting Effect Parameters");
                ImGui::Spacing();

                ImGui::Text("Radius");
                if (ImGui::SliderInt("##oil_radius", &params.oil_painting_radius, 1, 10)) {
                    if (image_loaded) apply_algorithm();
                }

                ImGui::Text("Intensity Levels");
                if (ImGui::SliderInt("##intensity_levels", &params.oil_painting_intensity_levels, 5, 50)) {
                    if (image_loaded) apply_algorithm();
                }
            }
            else if (current_algorithm == "Color Cast Detection") {
                ImGui::Text("🎨 Color Cast Detection Parameters");
                ImGui::Spacing();

                ImGui::Text("Threshold");
                if (ImGui::SliderFloat("##threshold", &params.color_cast_threshold, 0.05f, 0.5f, "%.2f")) {
                    if (image_loaded) apply_algorithm();
                }
            }

            ImGui::EndChild();
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Select an algorithm to see parameters");
        }
        ImGui::End();
    }

    // Bottom Panel - History
    if (show_history_panel) {
        ImGui::SetNextWindowPos(ImVec2(0, display_size.y - status_height - bottom_panel_height));
        ImGui::SetNextWindowSize(ImVec2(display_size.x * 0.5f, bottom_panel_height));
        ImGui::Begin("History", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (history.empty()) {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No history available");
        } else {
            for (int i = 0; i < history.size(); i++) {
                bool is_selected = (i == current_history_index);
                if (ImGui::Selectable(history[i].algorithm.c_str(), is_selected)) {
                    current_history_index = i;
                    processed_image = history[i].image.clone();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Click to restore this state");
                }
            }
        }
        ImGui::End();
    }

    // Bottom Panel - Info
    if (show_info_panel) {
        ImGui::SetNextWindowPos(ImVec2(display_size.x * 0.5f, display_size.y - status_height - bottom_panel_height));
        ImGui::SetNextWindowSize(ImVec2(display_size.x * 0.5f, bottom_panel_height));
        ImGui::Begin("Info", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (image_loaded) {
            ImGui::Text("Image Information:");
            ImGui::Separator();
            ImGui::Text("Dimensions: %dx%d", original_image.cols, original_image.rows);
            ImGui::Text("Channels: %d", original_image.channels());
            ImGui::Text("Type: %s", original_image.type() == CV_8UC3 ? "BGR" : "Grayscale");
            ImGui::Text("Memory: %.2f MB", (original_image.total() * original_image.elemSize()) / (1024.0f * 1024.0f));

            ImGui::Spacing();
            ImGui::Text("Current Algorithm: %s", current_algorithm.c_str());
            ImGui::Text("Zoom Level: %.1f%%", zoom_level * 100);

            if (!history.empty()) {
                ImGui::Spacing();
                ImGui::Text("History: %d/%d", current_history_index + 1, history.size());
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No image loaded");
        }
        ImGui::End();
    }

    // Main Image Display Area
    ImGui::SetNextWindowPos(ImVec2(left_panel_width, main_area_y));
    ImGui::SetNextWindowSize(ImVec2(display_size.x - left_panel_width - right_panel_width, main_area_height));
    ImGui::Begin("Image Display", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    if (image_loaded) {
        // Handle mouse input for zoom and pan
        ImVec2 mouse_pos = ImGui::GetMousePos();
        bool is_hovered = ImGui::IsWindowHovered();

        if (is_hovered && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            if (!is_panning) {
                is_panning = true;
                last_mouse_pos = mouse_pos;
            } else {
                ImVec2 delta = ImVec2(mouse_pos.x - last_mouse_pos.x, mouse_pos.y - last_mouse_pos.y);
                pan_offset.x += delta.x;
                pan_offset.y += delta.y;
                last_mouse_pos = mouse_pos;
            }
        } else {
            is_panning = false;
        }

        // Mouse wheel zoom
        float wheel = ImGui::GetIO().MouseWheel;
        if (is_hovered && wheel != 0) {
            zoom_level += wheel * 0.1f;
            zoom_level = max(0.1f, min(zoom_level, 5.0f));
        }

        // Calculate display area
        ImVec2 window_size = ImGui::GetWindowSize();
        ImVec2 content_size = ImVec2(window_size.x - 20, window_size.y - 80);

        // Calculate image display size with zoom
        float image_aspect = (float)original_image.cols / original_image.rows;
        float window_aspect = content_size.x / content_size.y;

        ImVec2 base_display_size;
        if (image_aspect > window_aspect) {
            base_display_size.x = content_size.x;
            base_display_size.y = content_size.x / image_aspect;
        } else {
            base_display_size.y = content_size.y;
            base_display_size.x = content_size.y * image_aspect;
        }

        ImVec2 display_size(base_display_size.x * zoom_level, base_display_size.y * zoom_level);

        // Center the image
        ImVec2 image_pos = ImGui::GetCursorScreenPos();
        image_pos.x += (content_size.x - display_size.x) * 0.5f + pan_offset.x;
        image_pos.y += (content_size.y - display_size.y) * 0.5f + pan_offset.y;

        // Display images side by side
        ImGui::BeginChild("ImageContainer", content_size, true);

        if (show_original) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Original Image");
            void* original_texture = mat_to_texture(original_image);
            ImGui::SetCursorScreenPos(image_pos);
            ImGui::Image(original_texture, display_size);

            if (show_processed && !processed_image.empty()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Processed Image");
                void* processed_texture = mat_to_texture(processed_image);
                ImGui::SetCursorScreenPos(ImVec2(image_pos.x + display_size.x + 20, image_pos.y));
                ImGui::Image(processed_texture, display_size);
            }
        } else if (show_processed && !processed_image.empty()) {
            ImGui::TextColored(ImVec4(0.0f, 0.7f, 1.0f, 1.0f), "Processed Image");
            void* processed_texture = mat_to_texture(processed_image);
            ImGui::SetCursorScreenPos(image_pos);
            ImGui::Image(processed_texture, display_size);
        }

        ImGui::EndChild();

        // Zoom info
        ImGui::Text("Zoom: %.1f%% | Pan: (%.0f, %.0f)", zoom_level * 100, pan_offset.x, pan_offset.y);
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No image loaded. Use File -> Open Image to start.");
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Or click the 'Open' button in the toolbar.");
    }
    ImGui::End();

    // Status Bar - Fixed at bottom
    ImGui::SetNextWindowPos(ImVec2(0, display_size.y - status_height));
    ImGui::SetNextWindowSize(ImVec2(display_size.x, status_height));
    ImGui::Begin("Status Bar", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

    if (image_loaded) {
        ImGui::Text("Image: %dx%d | Algorithm: %s | Zoom: %.1f%%",
                   original_image.cols, original_image.rows, current_algorithm.c_str(), zoom_level * 100);
    } else {
        ImGui::Text("Ready");
    }
    ImGui::End();
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        cout << "Failed to initialize GLFW" << endl;
        return -1;
    }

    // Create window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window = glfwCreateWindow(1600, 1000, "IP101 Image Processing Studio - Professional Edition", nullptr, nullptr);
    if (!window) {
        cout << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Setup professional style
    setup_professional_style();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        render_gui();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.08f, 0.08f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
