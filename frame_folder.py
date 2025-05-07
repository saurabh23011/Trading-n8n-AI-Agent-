#frame_folder
import easyocr
import cv2
import numpy as np
import os
from pathlib import Path

def extract_text_from_image(image_path):
    # Initialize the EasyOCR reader
    # Specify languages (e.g., ['en'] for English)
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    
    # Load and preprocess the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # Optional: Preprocess image for better OCR results
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR on the preprocessed image
    results = reader.readtext(thresh)
    
    # Extract and format the text
    extracted_text = []
    for (bbox, text, prob) in results:
        extracted_text.append({
            'text': text,
            'confidence': prob,
            'bounding_box': bbox
        })
    
    return extracted_text

def process_folder(folder_path):
    # Ensure folder exists
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder {folder_path} does not exist or is not a directory.")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Store results for all images
    all_results = {}
    
    # Iterate through all files in the folder
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            try:
                print(f"Processing: {file_path.name}")
                text_data = extract_text_from_image(file_path)
                all_results[file_path.name] = text_data
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
    
    return all_results

def save_results_to_file(results, output_file):
    # Save extracted text to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_name, text_data in results.items():
            f.write(f"\nImage: {image_name}\n")
            f.write("-" * 50 + "\n")
            if text_data:
                for item in text_data:
                    f.write(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}\n")
            else:
                f.write("No text detected.\n")

def main():
    # Example usage
    folder_path = 'frame'  # Replace with your folder path
    output_file = 'ocr_results.txt'  # Output file for results
    
    try:
        # Process all images in the folder
        results = process_folder(folder_path)
        
        # Save results to a file
        save_results_to_file(results, output_file)
        print(f"Results saved to {output_file}")
        
        # Optional: Print results to console
        for image_name, text_data in results.items():
            print(f"\nImage: {image_name}")
            print("-" * 50)
            if text_data:
                for item in text_data:
                    print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
            else:
                print("No text detected.")
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()