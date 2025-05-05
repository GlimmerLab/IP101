# Optimized Basic Image Processing Algorithms

English | [简体中文](README.md)

This directory contains C++ implementations of basic image processing algorithms. All implementations have been deeply optimized to achieve performance levels close to or matching OpenCV.

## 🚀 Build Requirements

- C++17 or higher
- OpenCV 4.x
- CPU with AVX2/SSE4.2 support
- OpenMP support
- CMake 3.10+

## 📦 Compilation Options

All files use the following basic compilation options:
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 [source_file] -o [output_file] `pkg-config --cflags --libs opencv4`
```

### File-Specific Compilation Options

1. **filtering.cpp** (Image Filtering)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 filtering.cpp -o filtering `pkg-config --cflags --libs opencv4`
```
Optimization focus:
- SIMD (AVX2/SSE4) vectorization
- OpenMP multi-threading
- Cache optimization and memory alignment
- Block processing
- Border pre-processing

2. **edge_detection.cpp** (Edge Detection)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 edge_detection.cpp -o edge_detection `pkg-config --cflags --libs opencv4`
```
Optimization focus:
- SIMD gradient computation
- Separable convolution
- Parallel processing
- Lookup table optimization
- Fixed-point arithmetic

3. **color_operations.cpp** (Color Operations)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 color_operations.cpp -o color_operations `pkg-config --cflags --libs opencv4`
```
Optimization focus:
- SIMD color conversion
- Parallel processing
- Lookup table optimization
- Memory access optimization
- Block processing

## 🔧 Optimization Strategies

### 1. SIMD Optimization
- AVX2 for bulk data processing
- SSE4.2 for specific algorithms
- Efficient use of vector instruction sets
- Memory alignment considerations

### 2. Memory Optimization
- Cache line alignment
- Minimized memory copies
- Optimized memory access patterns
- Block processing

### 3. Parallel Optimization
- OpenMP dynamic scheduling
- Adaptive thread count
- Task blocking
- Load balancing

### 4. Algorithm Optimization
- Lookup table pre-computation
- Fixed-point arithmetic
- Separable convolution
- Border pre-processing

### 5. Compilation Optimization
- Highest optimization level
- Auto-vectorization enabled
- Link-time optimization
- CPU-specific instruction sets

## 📊 Performance Comparison

Each implementation includes performance comparison tests with OpenCV. Test results show:
- Mean Filter: 80-90% of OpenCV performance
- Median Filter: 70-80% of OpenCV performance
- Gaussian Filter: 75-85% of OpenCV performance
- Edge Detection: 85-95% of OpenCV performance
- Color Conversion: 90-100% of OpenCV performance

## 🔍 Usage Examples

Each file contains complete example code and performance testing functions. To run examples:

```bash
# Compile
g++ -O3 -fopenmp -mavx2 -mfma -msse4.2 filtering.cpp -o filtering `pkg-config --cflags --libs opencv4`

# Run
./filtering
```

## 📝 Important Notes

1. Ensure your CPU supports the required instruction sets
2. Adjust thread count and block size based on hardware
3. Monitor memory usage for large image processing
4. Choose optimization strategies based on specific use cases

## 🔄 Changelog

- 2024-01: Initial version with basic optimizations
- 2024-01: Added SIMD optimizations
- 2024-01: Added performance testing framework
- 2024-01: Completed documentation