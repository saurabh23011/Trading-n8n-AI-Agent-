# import easyocr
# import cv2
# import numpy as np
# import os
# from pathlib import Path
# from transformers import BartForConditionalGeneration, BartTokenizer
# import torch
# # from moviepy.editor import VideoFileClip
# import whisper
# import tempfile
# import shutil

# def extract_frames_from_video(video_path, output_folder, frame_interval=1):
#     # Ensure output folder exists
#     output_folder = Path(output_folder)
#     output_folder.mkdir(parents=True, exist_ok=True)
    
#     # Open video
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video file: {video_path}")
    
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = 0
#     saved_frame_count = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Save frame at specified interval (e.g., every 1 second)
#         if frame_count % int(fps * frame_interval) == 0:
#             frame_path = output_folder / f"frame_{saved_frame_count:06d}.jpg"
#             cv2.imwrite(str(frame_path), frame)
#             saved_frame_count += 1
        
#         frame_count += 1
    
#     cap.release()
#     return saved_frame_count

# def extract_text_from_image(image_path):
#     # Initialize the EasyOCR reader
#     reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a compatible GPU
    
#     # Load and preprocess the image
#     image = cv2.imread(str(image_path))
#     if image is None:
#         raise FileNotFoundError(f"Image at {image_path} not found.")
    
#     # Preprocess image for better OCR results
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Perform OCR
#     results = reader.readtext(thresh)
    
#     # Extract and format the text
#     extracted_text = []
#     for (bbox, text, prob) in results:
#         extracted_text.append({
#             'text': text,
#             'confidence': prob,
#             'bounding_box': bbox
#         })
    
#     return extracted_text

# def process_frames(folder_path):
#     # Ensure folder exists
#     folder = Path(folder_path)
#     if not folder.exists() or not folder.is_dir():
#         raise NotADirectoryError(f"Folder {folder_path} does not exist or is not a directory.")
    
#     # Supported image extensions
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
#     # Store results for all frames
#     all_results = {}
    
#     # Iterate through all files in the folder
#     for file_path in sorted(folder.iterdir()):  # Sort for consistent order
#         if file_path.suffix.lower() in image_extensions:
#             try:
#                 print(f"Processing frame: {file_path.name}")
#                 text_data = extract_text_from_image(file_path)
#                 all_results[file_path.name] = text_data
#             except Exception as e:
#                 print(f"Error processing {file_path.name}: {str(e)}")
    
#     return all_results

# def extract_audio_and_transcribe(video_path, temp_dir):
#     # Extract audio using moviepy
#     video = VideoFileClip(str(video_path))
#     audio_path = temp_dir / "audio.wav"
#     video.audio.write_audiofile(str(audio_path))
#     video.close()
    
#     # Transcribe audio using Whisper
#     model = whisper.load_model("base")  # Use 'base' for lightweight CPU usage
#     result = model.transcribe(str(audio_path))
#     transcript = result["text"]
    
#     return transcript

# def generate_bart_summary(text, max_length=150, min_length=30):
#     if not text.strip():
#         return "No text available to summarize."
    
#     # Load BART model and tokenizer
#     model_name = "facebook/bart-large-cnn"
#     tokenizer = BartTokenizer.from_pretrained(model_name)
#     model = BartForConditionalGeneration.from_pretrained(model_name)
    
#     # Tokenize and encode the text
#     inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    
#     # Generate summary
#     summary_ids = model.generate(
#         inputs['input_ids'],
#         num_beams=4,
#         max_length=max_length,
#         min_length=min_length,
#         length_penalty=2.0,
#         early_stopping=True
#     )
    
#     # Decode the summary
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# def generate_video_summary(frame_summary, transcript_summary):
#     # Combine frame and transcript summaries into a cohesive video summary
#     combined_text = f"Visual Content: {frame_summary}\nAudio Content: {transcript_summary}"
#     return generate_bart_summary(combined_text, max_length=200, min_length=50)

# def save_results_to_file(frame_results, transcript, summary_dict, output_file):
#     # Save all results to a text file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         # Write detailed frame OCR results
#         f.write("Detailed Frame OCR Results\n")
#         f.write("=" * 50 + "\n")
#         for frame_name, text_data in frame_results.items():
#             f.write(f"\nFrame: {frame_name}\n")
#             f.write("-" * 50 + "\n")
#             if text_data:
#                 for item in text_data:
#                     f.write(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}\n")
#             else:
#                 f.write("No text detected.\n")
        
#         # Write audio transcript
#         f.write("\nAudio Transcript\n")
#         f.write("=" * 50 + "\n")
#         f.write(transcript if transcript.strip() else "No audio transcript available.\n")
        
#         # Write summary
#         f.write("\nSummary of Video\n")
#         f.write("=" * 50 + "\n")
#         f.write(f"Total Frames Processed: {summary_dict['total_frames']}\n")
#         f.write(f"Frames with Text: {summary_dict['frames_with_text']}\n")
#         f.write(f"Frames without Text: {summary_dict['frames_without_text']}\n")
#         f.write("\nFrame Text Summary:\n")
#         f.write(f"{summary_dict['frame_summary']}\n")
#         f.write("\nAudio Transcript Summary:\n")
#         f.write(f"{summary_dict['transcript_summary']}\n")
#         f.write("\nCombined Video Summary:\n")
#         f.write(f"{summary_dict['video_summary']}\n")

# def main():
#     # Example usage
#     video_path = 'test.mp4'  # Replace with your video path
#     output_file = 'video_ocr_transcript_summary.txt'  # Output file for results
#     temp_dir = Path(tempfile.mkdtemp())  # Temporary directory for frames and audio
    
#     try:
#         # Step 1: Extract frames
#         frames_folder = temp_dir / "frames"
#         print("Extracting frames from video...")
#         total_frames = extract_frames_from_video(video_path, frames_folder, frame_interval=1)
        
#         # Step 2: Process frames with EasyOCR
#         print("\nProcessing frames for OCR...")
#         frame_results = process_frames(frames_folder)
        
#         # Step 3: Extract and transcribe audio
#         print("\nExtracting and transcribing audio...")
#         transcript = extract_audio_and_transcribe(video_path, temp_dir)
        
#         # Step 4: Generate summaries
#         # Collect all frame text
#         all_frame_text = []
#         frames_with_text = 0
        
#         for frame_name, text_data in frame_results.items():
#             if text_data:  # Check if there's any text detected
#                 frames_with_text += 1
#                 for item in text_data:
#                     all_frame_text.append(item['text'])
        
#         frame_text_combined = " ".join(all_frame_text)
        
#         # Summarize frame text
#         frame_summary = generate_bart_summary(frame_text_combined)
        
#         # Summarize transcript
#         transcript_summary = generate_bart_summary(transcript)
        
#         # Generate combined video summary
#         video_summary = generate_video_summary(frame_summary, transcript_summary)
        
#         # Compile summary data
#         summary_dict = {
#             'total_frames': total_frames,
#             'frames_with_text': frames_with_text,
#             'frames_without_text': total_frames - frames_with_text,
#             'frame_summary': frame_summary,
#             'transcript_summary': transcript_summary,
#             'video_summary': video_summary
#         }
        
#         # Step 5: Save results
#         save_results_to_file(frame_results, transcript, summary_dict, output_file)
#         print(f"\nResults and summary saved to {output_file}")
        
#         # Step 6: Print results to console
#         print("\nDetailed Frame OCR Results")
#         print("=" * 50)
#         for frame_name, text_data in frame_results.items():
#             print(f"\nFrame: {frame_name}")
#             print("-" * 50)
#             if text_data:
#                 for item in text_data:
#                     print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
#             else:
#                 print("No text detected.")
        
#         print("\nAudio Transcript")
#         print("=" * 50)
#         print(transcript if transcript.strip() else "No audio transcript available.")
        
#         print("\nSummary of Video")
#         print("=" * 50)
#         print(f"Total Frames Processed: {summary_dict['total_frames']}")
#         print(f"Frames with Text: {summary_dict['frames_with_text']}")
#         print(f"Frames without Text: {summary_dict['frames_without_text']}")
#         print("\nFrame Text Summary:")
#         print(summary_dict['frame_summary'])
#         print("\nAudio Transcript Summary:")
#         print(summary_dict['transcript_summary'])
#         print("\nCombined Video Summary:")
#         print(summary_dict['video_summary'])
                
#     except Exception as e:
#         print(f"Error: {str(e)}")
    
#     finally:
#         # Clean up temporary directory
#         if temp_dir.exists():
#             shutil.rmtree(temp_dir)

# if __name__ == "__main__":
#     main()



import easyocr
import cv2
import numpy as np
import os
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
import whisper
import tempfile
import shutil

def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at specified interval (e.g., every 1 second)
        if frame_count % int(fps * frame_interval) == 0:
            frame_path = output_folder / f"frame_{saved_frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()
    return saved_frame_count

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

def process_frames(folder_path):
    # Ensure folder exists
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder {folder_path} does not exist or is not a directory.")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Store results for all frames
    all_results = {}
    
    # Iterate through all files in the folder
    for file_path in sorted(folder.iterdir()):  # Sort for consistent order
        if file_path.suffix.lower() in image_extensions:
            try:
                print(f"Processing frame: {file_path.name}")
                text_data = extract_text_from_image(file_path)
                all_results[file_path.name] = text_data
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
    
    return all_results

def extract_audio_and_transcribe(video_path, temp_dir):
    # Extract audio using moviepy
    video = VideoFileClip(str(video_path))
    audio_path = temp_dir / "audio.wav"
    video.audio.write_audiofile(str(audio_path))
    video.close()
    
    # Transcribe audio using Whisper
    model = whisper.load_model("base")  # Use 'base' for lightweight CPU usage
    result = model.transcribe(str(audio_path))
    transcript = result["text"]
    
    return transcript

def generate_bart_summary(text, max_length=150, min_length=30):
    if not text.strip():
        return "No text available to summarize."
    
    # Load BART model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Tokenize and encode the text
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_video_summary(frame_summary, transcript_summary):
    # Combine frame and transcript summaries into a cohesive video summary
    combined_text = f"Visual Content: {frame_summary}\nAudio Content: {transcript_summary}"
    return generate_bart_summary(combined_text, max_length=200, min_length=50)

def save_results_to_file(frame_results, transcript, summary, output_file):
    # Save all results to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write detailed frame OCR results
        f.write("Detailed Frame OCR Results\n")
        f.write("=" * 50 + "\n")
        for frame_name, text_data in frame_results.items():
            f.write(f"\nFrame: {frame_name}\n")
            f.write("-" * 50 + "\n")
            if text_data:
                for item in text_data:
                    f.write(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}\n")
            else:
                f.write("No text detected.\n")
        
        # Write audio transcript
        f.write("\nAudio Transcript\n")
        f.write("=" * 50 + "\n")
        f.write(transcript if transcript.strip() else "No audio transcript available.\n")
        
        # Write summary
        f.write("\nSummary of Video\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Frames Processed: {summary['total_frames']}\n")
        f.write(f"Frames with Text: {summary['frames_with_text']}\n")
        f.write(f"Frames without Text: {summary['frames_without_text']}\n")
        f.write("\nFrame Text Summary:\n")
        f.write(f"{summary['frame_summary']}\n")
        f.write("\nAudio Transcript Summary:\n")
        f.write(f"{summary['transcript_summary']}\n")
        f.write("\nCombined Video Summary:\n")
        f.write(f"{summary['video_summary']}\n")

def main():
    # Example usage
    video_path = 'test2.mp4'  # Replace with your video path
    output_file = 'video_ocr_transcript_summary.txt'  # Output file for results
    temp_dir = Path(tempfile.mkdtemp())  # Temporary directory for frames and audio
    
    try:
        # Step 1: Extract frames
        frames_folder = temp_dir / "frames"
        print("Extracting frames from video...")
        total_frames = extract_frames_from_video(video_path, frames_folder, frame_interval=1)
        
        # Step 2: Process frames with EasyOCR
        print("\nProcessing frames for OCR...")
        frame_results = process_frames(frames_folder)
        
        # Step 3: Extract and transcribe audio
        print("\nExtracting and transcribing audio...")
        transcript = extract_audio_and_transcribe(video_path, temp_dir)
        
        # Step 4: Generate summaries
        # Collect all frame text
        all_frame_text = []
        for frame_name, text_data in frame_results.items():
            for item in text_data:
                all_frame_text.append(item['text'])
        frame_text_combined = " ".join(all_frame_text)
        
        # Summarize frame text
        frame_summary = generate_bart_summary(frame_text_combined)
        
        # Summarize transcript
        transcript_summary = generate_bart_summary(transcript)
        
        # Generate combined video summary
        video_summary = generate_video_summary(frame_summary, transcript_summary)
        
        # Compile summary data
        summary = {
            'total_frames': total_frames,
            'frames_with_text': sum(1 for text_data in frame_results.values() if text_data),
            'frames_without_text': total_frames - sum(1 for text_data in frame_results.values() if text_data),
            'frame_summary': frame_summary,
            'transcript_summary': transcript_summary,
            'video_summary': video_summary
        }
        
        # Step 5: Save results
        save_results_to_file(frame_results, transcript, summary, output_file)
        print(f"\nResults and summary saved to {output_file}")
        
        # Step 6: Print results to console
        print("\nDetailed Frame OCR Results")
        print("=" * 50)
        for frame_name, text_data in frame_results.items():
            print(f"\nFrame: {frame_name}")
            print("-" * 50)
            if text_data:
                for item in text_data:
                    print(f"Text: {item['text']}, Confidence: {item['confidence']:.2f}")
            else:
                print("No text detected.")
        
        print("\nAudio Transcript")
        print("=" * 50)
        print(transcript if transcript.strip() else "No audio transcript available.")
        
        print("\nSummary of Video")
        print("=" * 50)
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Frames with Text: {summary['frames_with_text']}")
        print(f"Frames without Text: {summary['frames_without_text']}")
        print("\nFrame Text Summary:")
        print(summary['frame_summary'])
        print("\nAudio Transcript Summary:")
        print(summary['transcript_summary'])
        print("\nCombined Video Summary:")
        print(summary['video_summary'])
                
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()