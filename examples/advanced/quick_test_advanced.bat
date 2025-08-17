@echo off
echo ========================================
echo IP101 Advanced Algorithms Quick Test
echo ========================================

REM 检查是否存在测试图像
if not exist "assets\imori.jpg" (
    echo Error: Test image not found at assets\imori.jpg
    echo Please make sure you have a test image available
    pause
    exit /b 1
)

echo.
echo Testing Advanced Filtering Algorithms...
echo ----------------------------------------

echo Testing Guided Filter...
guided_filter_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Guided filter test failed
    pause
    exit /b 1
)

echo Testing Side Window Filter...
side_window_filter_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Side window filter test failed
    pause
    exit /b 1
)

echo Testing Homomorphic Filter...
homomorphic_filter_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Homomorphic filter test failed
    pause
    exit /b 1
)

echo.
echo Testing Image Correction Algorithms...
echo ----------------------------------------

echo Testing Automatic White Balance...
automatic_white_balance_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Automatic white balance test failed
    pause
    exit /b 1
)

echo Testing Gamma Correction...
gamma_correction_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Gamma correction test failed
    pause
    exit /b 1
)

echo Testing Auto Level...
auto_level_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Auto level test failed
    pause
    exit /b 1
)

echo Testing Backlight Correction...
backlight_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Backlight correction test failed
    pause
    exit /b 1
)

echo Testing Illumination Correction...
illumination_correction_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Illumination correction test failed
    pause
    exit /b 1
)

echo.
echo Testing Image Defogging Algorithms...
echo ----------------------------------------

echo Testing Dark Channel Defogging...
dark_channel_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Dark channel defogging test failed
    pause
    exit /b 1
)

echo Testing Realtime Dehazing...
realtime_dehazing_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Realtime dehazing test failed
    pause
    exit /b 1
)

echo Testing Median Filter Defogging...
median_filter_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Median filter defogging test failed
    pause
    exit /b 1
)

echo Testing Fast Defogging...
fast_defogging_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Fast defogging test failed
    pause
    exit /b 1
)

echo Testing Guided Filter Dehazing...
guided_filter_dehazing_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Guided filter dehazing test failed
    pause
    exit /b 1
)

echo.
echo Testing Image Enhancement Algorithms...
echo ----------------------------------------

echo Testing Retinex MSRCR...
retinex_msrcr_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Retinex MSRCR test failed
    pause
    exit /b 1
)

echo Testing Adaptive Logarithmic Mapping...
adaptive_logarithmic_mapping_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Adaptive logarithmic mapping test failed
    pause
    exit /b 1
)

echo Testing Automatic Color Equalization...
automatic_color_equalization_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Automatic color equalization test failed
    pause
    exit /b 1
)

echo Testing HDR...
hdr_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: HDR test failed
    pause
    exit /b 1
)

echo Testing Multi-scale Detail Enhancement...
multi_scale_detail_enhancement_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Multi-scale detail enhancement test failed
    pause
    exit /b 1
)

echo Testing Real-time Adaptive Contrast...
real_time_adaptive_contrast_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Real-time adaptive contrast test failed
    pause
    exit /b 1
)

echo.
echo Testing Image Effects Algorithms...
echo ----------------------------------------

echo Testing Cartoon Effect...
cartoon_effect_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Cartoon effect test failed
    pause
    exit /b 1
)

echo Testing Motion Blur Effect...
motion_blur_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Motion blur effect test failed
    pause
    exit /b 1
)

echo Testing Oil Painting Effect...
oil_painting_effect_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Oil painting effect test failed
    pause
    exit /b 1
)

echo Testing Skin Beauty Effect...
skin_beauty_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Skin beauty effect test failed
    pause
    exit /b 1
)

echo Testing Spherize Effect...
spherize_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Spherize effect test failed
    pause
    exit /b 1
)

echo Testing Unsharp Masking Effect...
unsharp_masking_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Unsharp masking effect test failed
    pause
    exit /b 1
)

echo Testing Vintage Effect...
vintage_effect_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Vintage effect test failed
    pause
    exit /b 1
)

echo.
echo Testing Special Detection Algorithms...
echo ----------------------------------------

echo Testing Rectangle Detection...
rectangle_detection_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Rectangle detection test failed
    pause
    exit /b 1
)

echo Testing Color Cast Detection...
color_cast_detection_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: Color cast detection test failed
    pause
    exit /b 1
)

echo Testing License Plate Detection...
license_plate_detection_test.exe assets\imori.jpg
if %errorlevel% neq 0 (
    echo ERROR: License plate detection test failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo All Advanced Algorithm Tests Completed!
echo ========================================
echo.
echo Test results have been saved to the current directory.
echo Check the generated image files to verify algorithm performance.
echo.

pause
