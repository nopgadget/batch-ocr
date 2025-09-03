# Batch OCR with TrOCR

A powerful batch OCR tool that leverages Microsoft's TrOCR (Transformer-based Optical Character Recognition) to transcribe PDF files and images into text files.

## Features

- **Batch Processing**: Process entire folders of PDF files and images at once
- **Multi-format Support**: Handles PDFs, PNG, JPG, JPEG, TIFF, BMP, and GIF files
- **High-Quality OCR**: Uses state-of-the-art TrOCR models from Hugging Face
- **GPU Acceleration**: Automatically uses GPU when available for faster processing
- **Robust PDF Handling**: Dual PDF processing methods (PyMuPDF + pdf2image fallback)
- **Command-line Interface**: Easy-to-use CLI with flexible options

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### System Dependencies

For PDF processing, you may need to install additional system dependencies:

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

## Usage

### Command Line Help

```
usage: main.py [-h] [--output-folder OUTPUT_FOLDER] [--model MODEL] [--verbose] input_folder

Batch OCR processing using TrOCR for PDFs and images

positional arguments:
  input_folder          Folder containing PDF files and images to process

options:
  -h, --help            show this help message and exit
  --output-folder OUTPUT_FOLDER
                        Output folder for transcribed text files (default: same as input)
  --model MODEL         TrOCR model to use (default: microsoft/trocr-base-handwritten)
  --verbose             Enable verbose logging
```

### Basic Usage

Process all PDF files and images in a folder:

```bash
python main.py /path/to/your/documents
```

### Advanced Usage Examples

```bash
# Specify output folder
python main.py /path/to/input --output-folder /path/to/output

# Use a different TrOCR model (for printed text)
python main.py /path/to/input --model microsoft/trocr-base-printed

# Use large model for better accuracy
python main.py /path/to/input --model microsoft/trocr-large-handwritten

# Enable verbose logging
python main.py /path/to/input --verbose

# Combine options
python main.py /path/to/input --output-folder /path/to/output --model microsoft/trocr-base-printed --verbose
```

### Available TrOCR Models

#### For Handwritten Text:
- **`microsoft/trocr-base-handwritten`** (default)
  - Size: ~334MB
  - Speed: Fast
  - Best for: Handwritten notes, forms, letters
  
- **`microsoft/trocr-large-handwritten`**
  - Size: ~1.3GB
  - Speed: Slower but more accurate
  - Best for: Complex handwritten documents requiring high accuracy

#### For Printed Text:
- **`microsoft/trocr-base-printed`**
  - Size: ~334MB
  - Speed: Fast
  - Best for: Typed documents, PDFs, books, articles
  
- **`microsoft/trocr-large-printed`**
  - Size: ~1.3GB
  - Speed: Slower but more accurate
  - Best for: Complex printed documents requiring high accuracy

#### Model Selection Guide:
- **For most PDFs and typed documents**: Use `microsoft/trocr-base-printed`
- **For handwritten content**: Use `microsoft/trocr-base-handwritten` (default)
- **For maximum accuracy**: Use the `large` variants (requires more memory/time)
- **For speed**: Use the `base` variants

## Output

For each input file, the script creates a corresponding `.txt` file with the full original filename:
- `document.pdf` → `document.pdf.txt`
- `image.jpg` → `image.jpg.txt`

PDF files are processed page by page, with page separators in the output text.

## Performance Tips

1. **Use GPU**: If you have a CUDA-compatible GPU, the script will automatically use it for faster processing
2. **Choose the right model**: Use printed text models for typed documents and handwritten models for handwritten content
3. **Batch size**: Process multiple files in one run for better efficiency

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **PDF processing errors**: Ensure Poppler is installed on your system
3. **Memory issues**: Use the base models instead of large models if you run out of memory

### GPU Support

To use GPU acceleration, ensure you have:
- CUDA-compatible GPU
- PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## License

This project uses TrOCR models from Microsoft Research. Please refer to their license terms for commercial usage.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool!
