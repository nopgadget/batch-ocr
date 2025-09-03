#!/usr/bin/env python3
"""
Batch OCR Script using TrOCR
Processes PDF files and images in a folder, transcribing each to a text file.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Union
import sys

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF for better PDF handling
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements with: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrOCRBatchProcessor:
    """Batch OCR processor using TrOCR for PDFs and images."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        """
        Initialize the TrOCR processor.
        
        Args:
            model_name: Hugging Face model name for TrOCR
        """
        logger.info(f"Loading TrOCR model: {model_name}")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        self.pdf_extensions = {'.pdf'}
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using TrOCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text as string
        """
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        # Decode generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    
    def process_pdf(self, pdf_path: Path) -> str:
        """
        Process a PDF file and extract text from all pages.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from all pages
        """
        logger.info(f"Processing PDF: {pdf_path}")
        all_text = []
        
        try:
            # Use PyMuPDF for better PDF handling
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                logger.info(f"Processing page {page_num + 1}/{len(doc)}")
                page = doc.load_page(page_num)
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("ppm")
                
                # Create PIL Image from bytes
                from io import BytesIO
                image = Image.open(BytesIO(img_data))
                
                # Extract text using TrOCR
                page_text = self.extract_text_from_image(image)
                if page_text:
                    all_text.append(f"--- Page {page_num + 1} ---")
                    all_text.append(page_text)
                    all_text.append("")  # Empty line between pages
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            # Fallback to pdf2image
            try:
                logger.info("Falling back to pdf2image conversion")
                pages = convert_from_path(pdf_path, dpi=200)
                
                for i, page in enumerate(pages):
                    logger.info(f"Processing page {i + 1}/{len(pages)} (fallback)")
                    page_text = self.extract_text_from_image(page)
                    if page_text:
                        all_text.append(f"--- Page {i + 1} ---")
                        all_text.append(page_text)
                        all_text.append("")
                        
            except Exception as fallback_e:
                logger.error(f"Fallback processing also failed: {fallback_e}")
                return f"Error processing PDF: {e}"
        
        return "\n".join(all_text)
    
    def process_image(self, image_path: Path) -> str:
        """
        Process an image file and extract text.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        logger.info(f"Processing image: {image_path}")
        
        try:
            image = Image.open(image_path)
            return self.extract_text_from_image(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return f"Error processing image: {e}"
    
    def process_file(self, file_path: Path) -> str:
        """
        Process a single file (PDF or image).
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text
        """
        extension = file_path.suffix.lower()
        
        if extension in self.pdf_extensions:
            return self.process_pdf(file_path)
        elif extension in self.image_extensions:
            return self.process_image(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return f"Unsupported file type: {extension}"
    
    def batch_process(self, input_folder: Path, output_folder: Path = None) -> None:
        """
        Process all supported files in a folder.
        
        Args:
            input_folder: Folder containing files to process
            output_folder: Folder to save output files (defaults to input_folder)
        """
        if output_folder is None:
            output_folder = input_folder
        
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        
        # Find all supported files
        supported_files = []
        for extension in self.image_extensions | self.pdf_extensions:
            supported_files.extend(input_folder.glob(f"*{extension}"))
            supported_files.extend(input_folder.glob(f"*{extension.upper()}"))
        
        if not supported_files:
            logger.warning(f"No supported files found in {input_folder}")
            return
        
        logger.info(f"Found {len(supported_files)} files to process")
        
        for i, file_path in enumerate(supported_files, 1):
            logger.info(f"Processing file {i}/{len(supported_files)}: {file_path.name}")
            
            # Extract text
            extracted_text = self.process_file(file_path)
            
            # Save to output file
            output_file = output_folder / f"{file_path.name}.txt"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                logger.info(f"Saved transcription to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving output file {output_file}: {e}")
        
        logger.info("Batch processing completed!")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch OCR processing using TrOCR for PDFs and images"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Folder containing PDF files and images to process"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Output folder for transcribed text files (default: same as input)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="""TrOCR model to use (default: microsoft/trocr-base-handwritten).
Available models:
  Handwritten text:
    - microsoft/trocr-base-handwritten (default, ~334MB, fast)
    - microsoft/trocr-large-handwritten (~1.3GB, more accurate)
  Printed text:
    - microsoft/trocr-base-printed (~334MB, fast, good for PDFs)
    - microsoft/trocr-large-printed (~1.3GB, more accurate)
Use printed models for typed documents/PDFs, handwritten models for handwritten content."""
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        logger.error(f"Input path is not a directory: {input_folder}")
        sys.exit(1)
    
    # Set output folder
    output_folder = Path(args.output_folder) if args.output_folder else input_folder
    
    # Initialize processor and run batch processing
    try:
        processor = TrOCRBatchProcessor(model_name=args.model)
        processor.batch_process(input_folder, output_folder)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
