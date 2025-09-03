#!/usr/bin/env python3
"""
Batch OCR Script with Multiple OCR Engines
Processes PDF files and images in a folder, transcribing each to a text file.
Supports EasyOCR (recommended for documents), Tesseract, and TrOCR engines.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Union
import sys
import time
from tqdm import tqdm

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    from pdf2image import convert_from_path
    import fitz  # PyMuPDF for better PDF handling
    from tqdm import tqdm
    import easyocr
    import pytesseract
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements with: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchOCRProcessor:
    """Batch OCR processor supporting multiple OCR engines for PDFs and images."""
    
    def __init__(self, ocr_engine: str = "easyocr", model_name: str = "microsoft/trocr-base-printed"):
        """
        Initialize the OCR processor.
        
        Args:
            ocr_engine: OCR engine to use ('easyocr', 'tesseract', or 'trocr')
            model_name: Hugging Face model name for TrOCR (only used if ocr_engine='trocr')
        """
        self.ocr_engine = ocr_engine.lower()
        
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        self.pdf_extensions = {'.pdf'}
        
        # Initialize the selected OCR engine
        if self.ocr_engine == "easyocr":
            logger.info("Initializing EasyOCR (supports 80+ languages)")
            self.reader = easyocr.Reader(['en'])  # Can add more languages: ['en', 'es', 'fr', etc.]
            logger.info("EasyOCR initialized successfully")
            
        elif self.ocr_engine == "tesseract":
            logger.info("Using Tesseract OCR")
            # Test if tesseract is available
            try:
                pytesseract.get_tesseract_version()
                logger.info("Tesseract OCR initialized successfully")
            except Exception as e:
                logger.error(f"Tesseract not found: {e}")
                logger.error("Please install tesseract: https://github.com/tesseract-ocr/tesseract")
                sys.exit(1)
                
        elif self.ocr_engine == "trocr":
            logger.info(f"Loading TrOCR model: {model_name}")
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            logger.info(f"TrOCR initialized on device: {self.device}")
            
        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}. Use 'easyocr', 'tesseract', or 'trocr'")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Ensure image is in RGB format first
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to OpenCV format (RGB to BGR)
            img_array = np.array(image)
            
            # Ensure we have a valid image array
            if img_array.dtype != np.uint8:
                img_array = img_array.astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            # Convert to grayscale
            if len(img_bgr.shape) == 3:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_bgr
            
            # Ensure gray is uint8
            if gray.dtype != np.uint8:
                gray = gray.astype(np.uint8)
            
            # Apply slight Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply adaptive thresholding for better text contrast
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(binary)
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original image")
            # Return original image if preprocessing fails
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image using the selected OCR engine.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text as string
        """
        try:
            if self.ocr_engine == "easyocr":
                # Ensure image is in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                # Convert PIL to numpy array for EasyOCR
                img_array = np.array(image)
                
                # Ensure the array is uint8
                if img_array.dtype != np.uint8:
                    img_array = img_array.astype(np.uint8)
                
                # EasyOCR works well with both original and preprocessed images
                # Try original first, then preprocessed if result is poor
                result = self.reader.readtext(img_array, paragraph=True, width_ths=0.7, height_ths=0.7)
                
                extracted_text = ''
                if result:
                    # Extract text from results
                    text_parts = []
                    for detection in result:
                        text = detection[1].strip()
                        confidence = detection[2]
                        # Only include text with reasonable confidence
                        if text and confidence > 0.3:
                            text_parts.append(text)
                    
                    extracted_text = '\n'.join(text_parts)
                
                # If result is poor (too short), try with preprocessed image
                if not extracted_text or len(extracted_text.strip()) < 10:
                    try:
                        logger.debug("Trying with preprocessed image for better results")
                        processed_image = self.preprocess_image(image)
                        img_array = np.array(processed_image)
                        
                        if img_array.dtype != np.uint8:
                            img_array = img_array.astype(np.uint8)
                            
                        result = self.reader.readtext(img_array, paragraph=True, width_ths=0.7, height_ths=0.7)
                        
                        if result:
                            text_parts = []
                            for detection in result:
                                text = detection[1].strip()
                                confidence = detection[2]
                                if text and confidence > 0.3:
                                    text_parts.append(text)
                            
                            preprocessed_text = '\n'.join(text_parts)
                            if len(preprocessed_text.strip()) > len(extracted_text.strip()):
                                extracted_text = preprocessed_text
                                
                    except Exception as preprocess_error:
                        logger.warning(f"Preprocessing failed: {preprocess_error}")
                
                return extracted_text if extracted_text else "No text detected"
            
            elif self.ocr_engine == "tesseract":
                # Preprocess image for better Tesseract results
                processed_image = self.preprocess_image(image)
                
                # Use Tesseract with custom config for better accuracy
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\'"()-'
                text = pytesseract.image_to_string(processed_image, config=custom_config)
                return text.strip()
            
            elif self.ocr_engine == "trocr":
                # TrOCR works best on cropped text regions, not full documents
                # This is kept for compatibility but not recommended for full documents
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return f"Error extracting text: {e}"
    
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
    
    def batch_process(self, input_folder: Path, output_folder: Path = None, override: bool = False) -> None:
        """
        Process all supported files in a folder.
        
        Args:
            input_folder: Folder containing files to process
            output_folder: Folder to save output files (defaults to input_folder)
            override: If True, process all files even if output exists
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
        
        # Filter out files that already have output unless override is specified
        files_to_process = []
        skipped_count = 0
        
        for file_path in supported_files:
            output_file = output_folder / f"{file_path.name}.txt"
            if output_file.exists() and not override:
                skipped_count += 1
                logger.debug(f"Skipping {file_path.name} - output already exists")
            else:
                files_to_process.append(file_path)
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} files with existing output (use --override to reprocess)")
        
        if not files_to_process:
            logger.info("No files to process (all outputs already exist)")
            return
        
        logger.info(f"Found {len(supported_files)} total files, processing {len(files_to_process)} files")
        
        # Initialize timing and statistics
        start_time = time.time()
        processed_count = 0
        total_chars = 0
        
        # Process files with progress bar
        with tqdm(files_to_process, desc="Processing files", unit="file") as pbar:
            for file_path in pbar:
                file_start_time = time.time()
                
                # Update progress bar description
                pbar.set_description(f"Processing {file_path.name}")
                
                # Extract text
                extracted_text = self.process_file(file_path)
                
                # Save to output file
                output_file = output_folder / f"{file_path.name}.txt"
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    
                    # Update statistics
                    processed_count += 1
                    total_chars += len(extracted_text)
                    file_time = time.time() - file_start_time
                    elapsed_time = time.time() - start_time
                    
                    # Calculate statistics
                    avg_time_per_file = elapsed_time / processed_count
                    files_per_sec = processed_count / elapsed_time
                    chars_per_sec = total_chars / elapsed_time
                    
                    # Update progress bar with statistics
                    pbar.set_postfix({
                        'files/sec': f'{files_per_sec:.2f}',
                        'chars/sec': f'{chars_per_sec:.0f}',
                        'avg_time': f'{avg_time_per_file:.1f}s'
                    })
                    
                    logger.debug(f"Saved transcription to: {output_file} ({len(extracted_text)} chars, {file_time:.1f}s)")
                    
                except Exception as e:
                    logger.error(f"Error saving output file {output_file}: {e}")
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info(f"\nBatch processing completed!")
        logger.info(f"Processed {processed_count} files in {total_time:.1f} seconds")
        logger.info(f"Average: {total_time/processed_count:.1f}s per file, {processed_count/total_time:.2f} files/sec")
        logger.info(f"Total characters extracted: {total_chars:,} ({total_chars/total_time:.0f} chars/sec)")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Batch OCR processing with multiple engine support for PDFs and images"
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
        "--engine",
        type=str,
        default="easyocr",
        choices=["easyocr", "tesseract", "trocr"],
        help="""OCR engine to use (default: easyocr).
Engines:
  easyocr: Best for documents, supports 80+ languages, good accuracy
  tesseract: Classic OCR, good for clean documents, requires tesseract installation
  trocr: Transformer-based, best for handwritten text or single lines"""
    )
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="""TrOCR model to use (only used with --engine trocr).
Available models:
  Handwritten text:
    - microsoft/trocr-base-handwritten (~334MB, fast)
    - microsoft/trocr-large-handwritten (~1.3GB, more accurate)
  Printed text:
    - microsoft/trocr-base-printed (default, ~334MB, fast, good for documents)
    - microsoft/trocr-large-printed (~1.3GB, more accurate)"""
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override existing output files and reprocess all files"
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
        processor = BatchOCRProcessor(ocr_engine=args.engine, model_name=args.model)
        processor.batch_process(input_folder, output_folder, override=args.override)
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
