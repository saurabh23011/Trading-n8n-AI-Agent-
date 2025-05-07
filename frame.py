import easyocr
import cv2
import numpy as np

def extract_text_from_image(image_path):
    # Initialize the EasyOCR reader
    # Specify languages (e.g., ['en'] for English)
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
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

def main():
    # Example usage
    image_path = 'image .jpg'  # Replace with your image path
    try:
        texts = extract_text_from_image(image_path)
        print("Extracted Text:")
        for item in texts:
            print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()