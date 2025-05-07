import easyocr
import cv2
import numpy as np
import os
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def extract_text_from_image(image_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    
    # Load and preprocess the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # Preprocess image for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR
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

def generate_bart_summary(text_list):
    if not text_list:
        return "No text available to summarize."
    
    # Combine all text
    combined_text = " ".join(text_list)
    
    # Load BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize and encode the text
    inputs = tokenizer(combined_text, max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_summary(results):
    total_frames = len(results)
    frames_with_text = sum(1 for text_data in results.values() if text_data)
    all_extracted_text = []
    
    for image_name, text_data in results.items():
        for item in text_data:
            all_extracted_text.append(item['text'])
    
    # Generate BART summary
    summary_text = generate_bart_summary(all_extracted_text)
    
    summary = {
        'total_frames': total_frames,
        'frames_with_text': frames_with_text,
        'frames_without_text': total_frames - frames_with_text,
        'all_text': all_extracted_text,
        'bart_summary': summary_text
    }
    
    return summary

def save_results_to_file(results, summary, output_file):
    # Save extracted text and summary to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write detailed results
        f.write("Detailed OCR Results\n")
        f.write("=" * 50 + "\n")
        for image_name, text_data in results.items():
            f.write(f"\nImage: {image_name}\n")
            f.write("-" * 50 + "\n")
            if text_data:
                for item in text_data:
                    f.write(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}\n")
            else:
                f.write("No text detected.\n")
        
        # Write summary
        f.write("\nSummary of All Frames\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Frames Processed: {summary['total_frames']}\n")
        f.write(f"Frames with Text: {summary['frames_with_text']}\n")
        f.write(f"Frames without Text: {summary['frames_without_text']}\n")
        f.write("All Extracted Text:\n")
        if summary['all_text']:
            for text in summary['all_text']:
                f.write(f"- {text}\n")
        else:
            f.write("No text extracted from any frame.\n")
        f.write("\nBART Summary:\n")
        f.write(f"{summary['bart_summary']}\n")

def main():
    # Example usage
    folder_path = 'frame'  # Replace with your folder path
    output_file = 'ocr_results_with_bart_summary.txt'  # Output file for results
    
    try:
        # Process all images in the folder
        results = process_folder(folder_path)
        
        # Generate summary
        summary = generate_summary(results)
        
        # Save results and summary to a file
        save_results_to_file(results, summary, output_file)
        print(f"Results and summary saved to {output_file}")
        
        # Print detailed results to console
        print("\nDetailed OCR Results")
        print("=" * 50)
        for image_name, text_data in results.items():
            print(f"\nImage: {image_name}")
            print("-" * 50)
            if text_data:
                for item in text_data:
                    print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
            else:
                print("No text detected.")
        
        # Print summary to console
        print("\nSummary of All Frames")
        print("=" * 50)
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Frames with Text: {summary['frames_with_text']}")
        print(f"Frames without Text: {summary['frames_without_text']}")
        print("All Extracted Text:")
        if summary['all_text']:
            for text in summary['all_text']:
                print(f"- {text}")
        else:
            print("No text extracted from any frame.")
        print("\nBART Summary:")
        print(summary['bart_summary'])
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()