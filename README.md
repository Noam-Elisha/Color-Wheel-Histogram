# Color Wheel Visualization Tool

A powerful and comprehensive Python tool for creating visual color histograms mapped onto color wheels, providing intuitive color analysis and visualization for digital images.

## 🎨 Overview

The Color Wheel tool analyzes images and creates beautiful color wheel visualizations where:
- **Colors are mapped to their natural position** on a traditional color wheel
- **Opacity represents frequency** - more common colors appear more opaque
- **Multiple output formats** including histograms, spectrums, and circular visualizations
- **High-performance processing** with GPU acceleration and parallel computing support

## ✨ Features

### Core Functionality
- 🎨 **Color Wheel Generation** - Maps image colors to traditional color wheel positions
- 📊 **Multiple Visualizations** - Histograms, spectrums, and circular color plots  
- 🖼️ **Flexible Image Support** - Works with all common image formats
- 🎛️ **Customizable Parameters** - Adjustable quantization, sampling, and output options

### Performance Optimizations
- ⚡ **GPU Acceleration** - CUDA support via CuPy for massive speedups
- 🚀 **JIT Compilation** - Numba acceleration for CPU-intensive operations
- 🔄 **Parallel Processing** - Multi-core processing for large images
- 💾 **Template Caching** - Smart caching system for wheel templates
- 🗄️ **Memory Mapping** - Efficient memory usage for large datasets

### Advanced Features
- 🎯 **Nearest Neighbor Matching** - Intelligent color mapping with KDTree optimization
- 📈 **Color Space Conversions** - RGB, HSV, and other color space support  
- 🎪 **Batch Processing** - Process multiple images efficiently
- 📋 **Command Line Interface** - Full CLI support with comprehensive options
- 🧪 **Comprehensive Testing** - 600+ tests ensuring reliability

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Color Wheel"

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux  
source .venv/bin/activate

# Install the package in development mode
pip install -e .

# Optional: Install GPU acceleration support
pip install cupy-cuda12x  # For CUDA 12.x
pip install numba scikit-learn  # For additional optimizations
```

### Basic Usage

```bash
# Generate color wheel from an image
python color_wheel.py input_image.jpg

# With custom output name
python color_wheel.py input_image.jpg --output my_color_wheel.png

# High-performance mode with GPU acceleration
python color_wheel.py input_image.jpg --use-gpu --use-numba --use-kdtree --parallel

# Batch processing
python color_wheel.py *.jpg --parallel
```

### Python API

```python
import color_wheel

# Generate color wheel with default settings
color_wheel.create_color_wheel('input_image.jpg', 'output_wheel.png')

# Advanced usage with custom parameters
result = color_wheel.create_color_wheel(
    input_path='image.jpg',
    output_path='wheel.png',
    quantization_factor=32,
    sample_factor=4,
    use_gpu=True,
    use_numba=True,
    parallel=True
)

# Create additional visualizations
color_wheel.create_opacity_histogram('image.jpg', 'histogram.png')
color_wheel.create_color_spectrum_histogram('image.jpg', 'spectrum.png')
```

## 📋 Command Line Options

```bash
python color_wheel.py INPUT [OPTIONS]

Required Arguments:
  INPUT                 Input image file path or glob pattern

Output Options:
  -o, --output OUTPUT   Output file path (default: auto-generated)
  --output-dir DIR      Output directory for batch processing

Processing Options:
  -q, --quantization N  Color quantization factor (default: 16)
  -s, --sample N        Pixel sampling factor (default: 1)
  --min-frequency N     Minimum color frequency threshold

Performance Options:
  --use-gpu            Enable GPU acceleration (requires CuPy)
  --use-numba          Enable JIT compilation (requires Numba)  
  --use-kdtree         Use KDTree for nearest neighbor (requires scikit-learn)
  --parallel           Enable parallel processing
  --no-mmap            Disable memory mapping

Visualization Options:
  --create-histogram   Generate opacity histogram
  --create-spectrum    Generate color spectrum
  --create-circular    Generate circular spectrum
  --wheel-size SIZE    Color wheel output size (default: 1000)

Advanced Options:
  --color-space SPACE  Color space for processing (rgb, hsv)
  --template-size SIZE Template resolution (default: 1000)
  --cache-dir DIR      Template cache directory
```

## 🎨 Visualization Types

### 1. Color Wheel
The main visualization showing colors mapped to their natural positions on a traditional color wheel with opacity indicating frequency.

### 2. Opacity Histogram  
A histogram showing color distribution with opacity-weighted bars.

### 3. Color Spectrum Histogram
Traditional histogram with colors represented by their actual RGB values.

### 4. Circular Color Spectrum
A circular arrangement showing color progression around the wheel perimeter.

## 🏗️ Architecture

### Core Components

- **`create_color_wheel()`** - Main color wheel generation function
- **`load_and_analyze_image()`** - Image loading and color analysis
- **`find_nearest_wheel_colors()`** - Color mapping to wheel positions
- **Template System** - Efficient wheel template generation and caching
- **Visualization Functions** - Multiple output format generators

### Optimization Systems

- **GPU Processing** - CuPy-based GPU acceleration for color operations
- **JIT Compilation** - Numba-accelerated hot paths for CPU processing
- **Parallel Computing** - Multi-process handling for large images
- **Smart Caching** - Template and computation result caching
- **Memory Mapping** - Efficient handling of large datasets

### Performance Characteristics

| Image Size | CPU Time | GPU Time | Memory Usage |
|------------|----------|----------|--------------|
| 1MP        | ~2s      | ~0.3s    | ~50MB       |
| 5MP        | ~8s      | ~0.8s    | ~200MB      |
| 10MP       | ~20s     | ~1.5s    | ~400MB      |
| 50MP       | ~120s    | ~6s      | ~2GB        |

## 🧪 Testing

The project includes a comprehensive test suite with 600+ tests covering all functionality:

```bash
# Run all tests
pytest

# Run with coverage report  
pytest --cov=color_wheel --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run performance tests
pytest tests/test_performance_edge_cases.py

# Use the custom test runner
python run_tests.py --coverage
```

### Test Categories
- **Core Functions** - Basic utility and color processing functions
- **Image Analysis** - Image loading and color extraction  
- **Template System** - Wheel template generation and caching
- **Nearest Neighbor** - Color matching algorithms
- **Visualizations** - All output format generation
- **CLI Interface** - Command-line argument parsing and integration
- **Performance** - Large image handling and optimization paths
- **Edge Cases** - Error conditions and boundary cases

## 🔧 Configuration

### Environment Variables
```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Select GPU device
export CUPY_CACHE_DIR=/path/to/cache  # CuPy cache location

# Performance tuning
export NUMBA_CACHE_DIR=/path/to/cache  # Numba cache location  
export OMP_NUM_THREADS=4  # OpenMP thread count
export OPENBLAS_NUM_THREADS=4  # BLAS thread count
```

### Template Caching
Templates are automatically cached in:
- **Windows**: `%APPDATA%/color_wheel/templates/`
- **macOS**: `~/Library/Caches/color_wheel/templates/`
- **Linux**: `~/.cache/color_wheel/templates/`

## 📊 Performance Optimization

### For Small Images (< 1MP)
```bash
python color_wheel.py image.jpg --quantization 8
```

### For Medium Images (1-10MP)  
```bash
python color_wheel.py image.jpg --use-numba --quantization 16 --sample 2
```

### For Large Images (> 10MP)
```bash
python color_wheel.py image.jpg --use-gpu --use-numba --use-kdtree --parallel --quantization 32 --sample 4
```

### Batch Processing
```bash
python color_wheel.py *.jpg --parallel --output-dir results/
```

## 🔍 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'color_wheel'**
```bash
# Install in development mode
pip install -e .
```

**CUDA/GPU Errors**
```bash
# Check GPU availability
python -c "import cupy; print('GPU available')"

# Fallback to CPU
python color_wheel.py image.jpg  # (without --use-gpu)
```

**Memory Issues**
```bash
# Reduce memory usage
python color_wheel.py image.jpg --quantization 8 --sample 4 --no-mmap
```

**Slow Performance**
```bash
# Enable all optimizations
python color_wheel.py image.jpg --use-gpu --use-numba --use-kdtree --parallel
```

### Performance Tips

1. **Use GPU acceleration** for large images when available
2. **Enable parallel processing** for multi-core systems  
3. **Increase sampling factor** for very large images to reduce processing time
4. **Use appropriate quantization** - higher values = faster processing, lower quality
5. **Keep templates cached** - don't clear cache directory for better performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python run_tests.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 Dependencies

### Required
- **Python 3.8+**
- **NumPy** - Numerical computing
- **OpenCV** - Image processing  
- **Matplotlib** - Visualization and plotting
- **Pillow** - Additional image format support

### Optional (for enhanced performance)
- **CuPy** - GPU acceleration
- **Numba** - JIT compilation
- **scikit-learn** - KDTree nearest neighbor optimization
- **psutil** - Memory monitoring

### Development
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **pytest-xdist** - Parallel test execution

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Traditional color theory and color wheel design principles
- The NumPy and OpenCV communities for excellent documentation
- Performance optimization techniques from the scientific Python ecosystem
- Testing methodologies from the pytest community

---

**Made with ❤️ and lots of ☕ by a passionate developer who loves color theory and data visualization**