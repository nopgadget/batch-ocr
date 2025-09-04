# Batch OCR with Multiple Engine Support

A powerful batch OCR tool that supports multiple OCR engines including EasyOCR (recommended), Tesseract, and Microsoft's TrOCR to transcribe PDF files and images into text files. Choose the best engine for your specific use case.

## Features

- **Multiple OCR Engines**: Choose between EasyOCR (recommended), Tesseract, or TrOCR based on your needs
- **Batch Processing**: Process entire folders of PDF files and images at once
- **Multi-format Support**: Handles PDFs, PNG, JPG, JPEG, TIFF, BMP, and GIF files
- **Smart Engine Selection**: EasyOCR for documents, TrOCR for handwritten text, Tesseract for clean printed text
- **Advanced Image Preprocessing**: Automatic image enhancement for better OCR accuracy
- **Robust PDF Handling**: Dual PDF processing methods (PyMuPDF + pdf2image fallback)
- **Progress Tracking**: Real-time progress bars with processing statistics
- **GPU Acceleration**: Automatically uses GPU when available (TrOCR engine)
- **Multi-language Support**: EasyOCR supports 80+ languages
- **Skip Processed Files**: Intelligent file skipping to avoid reprocessing existing outputs
- **Command-line Interface**: Easy-to-use CLI with flexible options

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### System Dependencies

#### For PDF Processing (Required):
**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
Download and install Poppler from: https://poppler.freedesktop.org/

#### For Tesseract Engine (Optional):
Only needed if you plan to use `--engine tesseract`

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

#### GPU Support (Optional):
For faster TrOCR processing, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line Help

```
usage: main.py [-h] [--output-folder OUTPUT_FOLDER] [--engine {easyocr,tesseract,trocr}] 
               [--model MODEL] [--override] [--verbose] input_folder

Batch OCR processing with multiple engine support for PDFs and images

positional arguments:
  input_folder          Folder containing PDF files and images to process

options:
  -h, --help            show this help message and exit
  --output-folder OUTPUT_FOLDER
                        Output folder for transcribed text files (default: same as input)
  --engine {easyocr,tesseract,trocr}
                        OCR engine to use (default: easyocr)
  --model MODEL         TrOCR model to use (only used with --engine trocr)
  --override            Override existing output files and reprocess all files
  --verbose             Enable verbose logging
```

### Basic Usage

Process all PDF files and images in a folder using EasyOCR (recommended):

```bash
python main.py /path/to/your/documents
```

### Engine-Specific Usage Examples

```bash
# Use EasyOCR (default, recommended for most documents)
python main.py /path/to/input --engine easyocr

# Use Tesseract OCR for clean printed documents
python main.py /path/to/input --engine tesseract

# Use TrOCR for handwritten text
python main.py /path/to/input --engine trocr --model microsoft/trocr-base-handwritten

# Use TrOCR for printed text with better accuracy
python main.py /path/to/input --engine trocr --model microsoft/trocr-large-printed
```

### Advanced Usage Examples

```bash
# Specify output folder
python main.py /path/to/input --output-folder /path/to/output

# Override existing files and reprocess everything
python main.py /path/to/input --override

# Enable verbose logging
python main.py /path/to/input --verbose

# Combine options
python main.py /path/to/input --output-folder /path/to/output --engine easyocr --verbose
```

## OCR Engine Selection Guide

### 🚀 EasyOCR (Recommended)
**Best for: Most documents, multi-language support, balanced accuracy and speed**

- ✅ **Excellent for documents**: Works great with PDFs, scanned documents, photos
- ✅ **80+ languages supported**: Automatic language detection
- ✅ **No additional setup**: Works out of the box
- ✅ **Good accuracy**: High-quality results for most content types
- ✅ **Reasonable speed**: Fast processing with good results
- ❌ **Memory usage**: Uses more RAM than Tesseract

**Use when**: Processing mixed documents, need multi-language support, want good results without configuration

### 📝 Tesseract OCR
**Best for: Clean printed text, simple documents, minimal resource usage**

- ✅ **Excellent for clean text**: Best results on high-quality scanned documents
- ✅ **Lightweight**: Lower memory usage
- ✅ **Fast**: Quick processing for simple documents
- ✅ **Mature**: Well-established, reliable engine
- ❌ **Setup required**: Needs separate installation
- ❌ **Limited with poor quality**: Struggles with low-quality or noisy images
- ❌ **Less robust**: May fail on complex layouts

**Use when**: Processing clean PDFs, need minimal resource usage, have high-quality scanned documents

### 🎯 TrOCR (Transformer-based)
**Best for: Handwritten text, complex layouts, maximum accuracy (when properly configured)**

- ✅ **Excellent for handwriting**: State-of-the-art handwritten text recognition
- ✅ **AI-powered**: Transformer architecture for complex text understanding
- ✅ **GPU acceleration**: Can leverage CUDA for faster processing
- ✅ **Highest accuracy**: For supported content types
- ❌ **Large models**: Significant disk space and memory requirements
- ❌ **Slower**: Especially without GPU acceleration
- ❌ **Best for text regions**: Designed for cropped text, not full documents

**Use when**: Processing handwritten content, need maximum accuracy, have GPU available

### Available TrOCR Models

#### For Handwritten Text:
- **`microsoft/trocr-base-handwritten`**
  - Size: ~334MB, Speed: Fast
  - Best for: Handwritten notes, forms, letters
  
- **`microsoft/trocr-large-handwritten`**
  - Size: ~1.3GB, Speed: Slower but more accurate
  - Best for: Complex handwritten documents requiring high accuracy

#### For Printed Text:
- **`microsoft/trocr-base-printed`** (default for TrOCR)
  - Size: ~334MB, Speed: Fast
  - Best for: Typed documents, PDFs, books, articles
  
- **`microsoft/trocr-large-printed`**
  - Size: ~1.3GB, Speed: Slower but more accurate
  - Best for: Complex printed documents requiring high accuracy

### Quick Selection Guide:
- **📄 Most PDFs and documents**: `--engine easyocr` (default)
- **✍️ Handwritten content**: `--engine trocr --model microsoft/trocr-base-handwritten`
- **📖 Clean printed text**: `--engine tesseract`
- **🎯 Maximum accuracy**: `--engine trocr --model microsoft/trocr-large-*`
- **⚡ Speed priority**: `--engine tesseract` or `--engine easyocr`

## Output

For each input file, the script creates a corresponding `.txt` file with the full original filename:
- `document.pdf` → `document.pdf.txt`
- `image.jpg` → `image.jpg.txt`

PDF files are processed page by page, with page separators in the output text. The script intelligently skips files that have already been processed unless you use the `--override` flag.

## Performance Tips

### Engine-Specific Optimization

#### EasyOCR (Recommended)
- **Memory**: Allocate 2-4GB RAM for best performance
- **Batch processing**: Process multiple files in one run for better efficiency
- **Image quality**: Works well with various image qualities without preprocessing

#### Tesseract OCR
- **Clean documents**: Best performance on high-quality, clean scanned documents
- **Preprocessing**: Images are automatically preprocessed for better results
- **Speed**: Fastest for simple, clean text documents
- **Memory**: Lowest memory usage among all engines

#### TrOCR
- **GPU acceleration**: Use CUDA-compatible GPU for 3-5x speed improvement
- **Model selection**: Use `base` models for speed, `large` models for accuracy
- **Memory requirements**: 
  - Base models: ~2GB RAM (8GB recommended)
  - Large models: ~6GB RAM (16GB recommended)
- **Document type**: Choose handwritten vs printed models appropriately

### General Tips
1. **Skip processed files**: Use default behavior to avoid reprocessing (saves significant time)
2. **Batch processing**: Process entire folders at once for better efficiency
3. **Output organization**: Use `--output-folder` to keep source and output separate
4. **Monitor progress**: Use `--verbose` flag to see detailed processing statistics

## Troubleshooting

### Installation Issues

**ImportError or missing dependencies:**
```bash
pip install -r requirements.txt
```
If specific packages fail, try installing them individually.

**PDF processing errors:**
- Ensure Poppler is installed on your system (see Installation section)
- On Windows, make sure Poppler is in your PATH

**Tesseract not found (when using `--engine tesseract`):**
- Install Tesseract OCR following the Installation section
- On Windows, ensure tesseract.exe is in your PATH

### Runtime Issues

#### EasyOCR Issues
- **High memory usage**: Normal behavior, ensure you have 2-4GB available RAM
- **Slow first run**: EasyOCR downloads models on first use (~100MB)
- **Language detection errors**: Specify languages manually: `Reader(['en', 'es'])` in code

#### Tesseract Issues
- **Poor results**: Works best with clean, high-contrast documents
- **Unrecognized text**: Try the EasyOCR engine for better accuracy
- **Configuration errors**: The script uses optimal settings automatically

#### TrOCR Issues
- **Memory errors**: 
  - Use base models instead of large models
  - Ensure 8GB+ RAM for base models, 16GB+ for large models
  - Enable GPU acceleration to reduce CPU memory usage
- **Slow processing**: 
  - Install CUDA-compatible PyTorch for GPU acceleration
  - Consider using EasyOCR for similar accuracy with better speed
- **Model download failures**: 
  - Check internet connection
  - Ensure sufficient disk space (~1.3GB for large models)

### Performance Issues

**Slow processing:**
- Use `--engine tesseract` for fastest processing of clean documents
- Use `--engine easyocr` for balanced speed and accuracy
- Enable GPU for TrOCR: install CUDA-compatible PyTorch

**Out of memory:**
- Use Tesseract engine for minimal memory usage
- Use TrOCR base models instead of large models
- Process smaller batches

**Poor OCR accuracy:**
- Try EasyOCR engine for general documents
- Use TrOCR for handwritten content
- Ensure good image quality (300+ DPI for scanned documents)

### GPU Support

For GPU acceleration with TrOCR:
- CUDA-compatible GPU (GTX 1060+ or RTX series recommended)
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

## Example Output

### Processing Statistics
The tool provides real-time progress tracking:
```
Processing files: 100%|████████████| 45/45 [02:34<00:00,  3.14files/sec, avg_time=2.1s, chars/sec=8.2k]

Batch processing completed!
Processed 45 files in 154.2 seconds
Average: 3.4s per file, 0.29 files/sec
Total characters extracted: 1,267,543 (8,223 chars/sec)
```

### Output Files
```
input/
├── document1.pdf
├── scanned_page.jpg
└── handwritten_note.png

output/
├── document1.pdf.txt
├── scanned_page.jpg.txt
└── handwritten_note.png.txt
```

## License

This project uses models from various sources:
- **TrOCR models**: Microsoft Research - refer to their license terms for commercial usage
- **EasyOCR**: Apache License 2.0
- **Tesseract**: Apache License 2.0

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool! Contributions are welcome for:
- Additional OCR engine support
- Performance optimizations
- UI improvements
- Documentation enhancements
