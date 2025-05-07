#app.py

import streamlit as st
import os
import time
from datetime import timedelta
import tempfile
import io
import base64
import pandas as pd
import numpy as np
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import cv2
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .result-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .timestamp {
        font-weight: bold;
        color: #0066cc;
    }
    .description {
        margin-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>YouTube Video Analyzer</h1>", unsafe_allow_html=True)
st.markdown("Analyze YouTube videos - extract frames, transcripts, and generate summaries.")

# Initialize session state variables if they don't exist
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'video_title' not in st.session_state:
    st.session_state.video_title = None
if 'video_thumbnail' not in st.session_state:
    st.session_state.video_thumbnail = None
if 'frame_images' not in st.session_state:
    st.session_state.frame_images = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'error' not in st.session_state:
    st.session_state.error = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Functions from the original code with adjustments
def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    youtube_url = youtube_url.strip()
    if "youtu.be" in youtube_url:
        return youtube_url.split("/")[-1].split("?")[0]
    elif "youtube.com" in youtube_url:
        if "v=" in youtube_url:
            return youtube_url.split("v=")[1].split("&")[0]
    raise ValueError("Could not extract YouTube video ID from the provided URL")

def get_transcript_with_timestamps(video_id):
    """Get the transcript with timestamps from a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        st.error(f"Error retrieving transcript: {e}")
        return None

def download_video(youtube_url, progress_bar=None):
    """Download a YouTube video for processing."""
    try:
        yt = YouTube(youtube_url)
        video = yt.streams.filter(progressive=True, file_extension="mp4").first()
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.close()
        
        # Download the video
        video.download(filename=temp_file.name)
        
        # Get the video thumbnail
        thumbnail_url = yt.thumbnail_url
        
        return temp_file.name, yt.title, thumbnail_url
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None, None, None

def extract_frames_and_convert(video_path, interval_seconds=60):
    """Extract frames from the video at regular intervals and convert to web-friendly format."""
    frames = []
    frame_timestamps = []
    frame_images_b64 = []
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame positions at specified intervals
    interval_frames = interval_seconds * fps
    
    for i in range(0, total_frames, int(interval_frames)):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            timestamp = timedelta(seconds=i/fps)
            frames.append(frame)
            frame_timestamps.append(str(timestamp).split('.')[0])  # Format: HH:MM:SS
            
            # Convert frame to base64 for web display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            
            # Resize for web display (optional)
            pil_img = pil_img.resize((400, 225))
            
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            frame_images_b64.append(f"data:image/jpeg;base64,{img_str}")
    
    video.release()
    return frames, frame_timestamps, frame_images_b64

def describe_frame(frame):
    """Describe content in a video frame using basic image analysis."""
    # Simple image analysis (without ML models)
    # This is a placeholder - in a production app, you would use proper image analysis
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb_frame.shape
    
    # Calculate brightness
    brightness = np.mean(rgb_frame)
    
    # Calculate color dominance
    avg_color = np.mean(rgb_frame, axis=(0, 1))
    r, g, b = avg_color
    
    # Determine dominant color
    max_val = max(r, g, b)
    if max_val == r:
        dominant = "red"
    elif max_val == g:
        dominant = "green"
    else:
        dominant = "blue"
    
    # Create a basic description
    if brightness < 50:
        brightness_desc = "dark"
    elif brightness < 150:
        brightness_desc = "medium brightness"
    else:
        brightness_desc = "bright"
    
    description = f"A {brightness_desc} {width}x{height} image with {dominant} tones."
    return description

def find_transcript_segment_for_timestamp(transcript, timestamp_str):
    """Find the transcript segment that corresponds to a specific timestamp."""
    if not transcript:
        return "No transcript available"
        
    timestamp_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp_str.split(":"))))
    
    relevant_segments = []
    for segment in transcript:
        start_time = segment['start']
        end_time = start_time + segment['duration']
        
        # If the timestamp falls within this segment or is close to it
        if start_time - 5 <= timestamp_seconds <= end_time + 5:
            relevant_segments.append(segment['text'])
    
    return " ".join(relevant_segments) if relevant_segments else "No transcript found for this timestamp"

def generate_simple_summary(transcript):
    """Generate a basic summary from the transcript."""
    if not transcript:
        return "No transcript available to summarize."
    
    # Get all transcript text
    full_text = " ".join([segment["text"] for segment in transcript])
    
    # For a very simple summary, just take the first ~300 characters
    if len(full_text) <= 300:
        return full_text
    
    # Find the nearest sentence end after ~300 characters
    end_pos = 300
    while end_pos < len(full_text) and end_pos < 350:
        if full_text[end_pos] in ['.', '!', '?']:
            end_pos += 1
            break
        end_pos += 1
    
    simple_summary = full_text[:end_pos]
    return simple_summary + "... (Summary is based on the beginning of the transcript)"

def read_urls_from_uploaded_file(uploaded_file):
    """Read YouTube URLs from an uploaded file."""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        urls = [line.strip() for line in content.split('\n') if line.strip()]
        return urls
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return []

def process_video(youtube_url, progress_bar):
    """Process a single YouTube video and return analysis results."""
    try:
        st.session_state.error = None
        video_id = extract_video_id(youtube_url)
        
        # Update progress
        progress_bar.progress(0.1)
        st.session_state.progress = 0.1
        
        # Get video transcript
        with st.spinner("Retrieving transcript..."):
            transcript = get_transcript_with_timestamps(video_id)
            
            if not transcript:
                st.warning("Could not retrieve transcript. The video might not have captions.")
                # Continue without transcript
        
        # Update progress
        progress_bar.progress(0.3)
        st.session_state.progress = 0.3
        
        # Download the video
        with st.spinner("Downloading video for processing..."):
            video_path, video_title, thumbnail_url = download_video(youtube_url)
            st.session_state.video_title = video_title
            st.session_state.video_thumbnail = thumbnail_url
            
            if not video_path:
                st.error("Could not download the video.")
                st.session_state.error = "Failed to download video"
                return None
        
        # Update progress
        progress_bar.progress(0.5)
        st.session_state.progress = 0.5
        
        # Extract frames at regular intervals
        with st.spinner("Extracting and analyzing frames..."):
            frames, frame_timestamps, frame_images_b64 = extract_frames_and_convert(video_path)
            st.session_state.frame_images = frame_images_b64
            
            # Process each frame
            frame_analysis = []
            
            for i, (frame, timestamp) in enumerate(zip(frames, frame_timestamps)):
                # Update progress for each frame
                current_progress = 0.5 + 0.3 * ((i + 1) / len(frames))
                progress_bar.progress(current_progress)
                st.session_state.progress = current_progress
                
                # Basic visual analysis
                visual_description = describe_frame(frame)
                
                # Get transcript at this timestamp
                transcript_segment = find_transcript_segment_for_timestamp(transcript, timestamp)
                
                frame_analysis.append({
                    "timestamp": timestamp,
                    "visual_description": visual_description,
                    "transcript": transcript_segment
                })
        
        # Generate simple summary
        with st.spinner("Generating summary..."):
            summary = generate_simple_summary(transcript)
        
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Update progress
        progress_bar.progress(1.0)
        st.session_state.progress = 1.0
        
        return {
            "title": video_title,
            "summary": summary,
            "frame_analysis": frame_analysis,
            "thumbnail_url": thumbnail_url
        }
    
    except Exception as e:
        st.error(f"Error processing video {youtube_url}: {e}")
        st.session_state.error = str(e)
        return None

# Sidebar for input options
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=80)
    st.header("Input Options")
    
    input_option = st.radio("Select input method:", ["YouTube URL", "Upload URLs File"])
    
    if input_option == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
        process_single = st.button("Analyze Video")
    else:
        uploaded_file = st.file_uploader("Upload a file with YouTube URLs (one per line)", type=["txt"])
        process_file = st.button("Process URLs from File")
    
    st.markdown("---")
    st.markdown("### Analysis Options")
    frame_interval = st.slider("Frame Interval (seconds)", min_value=30, max_value=300, value=60, step=30)

# Main content
if input_option == "YouTube URL" and process_single and youtube_url:
    st.session_state.processing = True
    st.session_state.analysis_results = None
    st.session_state.error = None
    progress_bar = st.progress(0)
    
    # Process the video
    results = process_video(youtube_url, progress_bar)
    
    if results:
        st.session_state.analysis_results = results
    
    st.session_state.processing = False

elif input_option == "Upload URLs File" and process_file and uploaded_file is not None:
    urls = read_urls_from_uploaded_file(uploaded_file)
    
    if not urls:
        st.error("No valid URLs found in the uploaded file.")
    else:
        st.write(f"Found {len(urls)} URLs in the file. Processing first URL...")
        
        st.session_state.processing = True
        st.session_state.analysis_results = None
        st.session_state.error = None
        progress_bar = st.progress(0)
        
        # Process the first video
        results = process_video(urls[0], progress_bar)
        
        if results:
            st.session_state.analysis_results = results
            
            # Store the remaining URLs for potential future processing
            if len(urls) > 1:
                st.session_state.remaining_urls = urls[1:]
                st.info(f"Processed first URL. {len(urls)-1} URLs remaining.")
        
        st.session_state.processing = False

# Display results if available
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Create columns for video info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.session_state.video_thumbnail:
            st.image(st.session_state.video_thumbnail, width=320)
    
    with col2:
        st.markdown(f"<h2>{results['title']}</h2>", unsafe_allow_html=True)
        
    # Summary Section
    st.markdown("<h3 class='sub-header'>Video Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-section'>{results['summary']}</div>", unsafe_allow_html=True)
    
    # Frame Analysis Section
    st.markdown("<h3 class='sub-header'>Frame-by-Frame Analysis</h3>", unsafe_allow_html=True)
    
    # Create tabs for different view options
    tab1, tab2 = st.tabs(["Visual Timeline", "Detailed List"])
    
    with tab1:
        for i, (analysis, img_b64) in enumerate(zip(results['frame_analysis'], st.session_state.frame_images)):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"<p class='timestamp'>Timestamp: {analysis['timestamp']}</p>", unsafe_allow_html=True)
                st.markdown(f"<img src='{img_b64}' width='320'>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<strong>Visual Description:</strong>", unsafe_allow_html=True)
                st.markdown(f"<p class='description'>{analysis['visual_description']}</p>", unsafe_allow_html=True)
                
                st.markdown("<strong>Transcript at this moment:</strong>", unsafe_allow_html=True)
                st.markdown(f"<p class='description'>{analysis['transcript'] if analysis['transcript'] else 'No transcript available for this timestamp.'}</p>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
    
    with tab2:
        # Create a DataFrame for easier display
        data = []
        for analysis in results['frame_analysis']:
            data.append({
                'Timestamp': analysis['timestamp'],
                'Visual Description': analysis['visual_description'],
                'Transcript': analysis['transcript'] if analysis['transcript'] else 'N/A'
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    # Export options
    st.markdown("<h3 class='sub-header'>Export Options</h3>", unsafe_allow_html=True)
    
    # Create a downloadable text file
    export_text = f"ANALYSIS OF: {results['title']}\n"
    export_text += "=" * 80 + "\n\n"
    export_text += "VIDEO SUMMARY:\n"
    export_text += "-" * 80 + "\n"
    export_text += results['summary'] + "\n\n"
    export_text += "FRAME-BY-FRAME ANALYSIS:\n"
    export_text += "-" * 80 + "\n\n"
    
    for analysis in results['frame_analysis']:
        export_text += f"TIMESTAMP: {analysis['timestamp']}\n"
        export_text += f"VISUAL: {analysis['visual_description']}\n"
        export_text += f"TRANSCRIPT: {analysis['transcript']}\n"
        export_text += "-" * 40 + "\n\n"
    
    # Convert to downloadable format
    b64 = base64.b64encode(export_text.encode()).decode()
    safe_title = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in results['title'])
    filename = f"{safe_title}_analysis.txt"
    
    st.download_button(
        label="Download Analysis as Text File",
        data=export_text,
        file_name=filename,
        mime="text/plain"
    )
    
    # CSV export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"{safe_title}_analysis.csv",
        mime="text/csv"
    )

elif st.session_state.processing:
    st.info("Processing video... Please wait.")
    
elif st.session_state.error:
    st.error(f"An error occurred: {st.session_state.error}")
    st.warning("Please try another URL or check your input.")

# Display instructions when no analysis is being shown
if not st.session_state.analysis_results and not st.session_state.processing:
    st.markdown("""
    ## How to Use This Tool
    
    1. **Choose your input method** in the sidebar:
       - Enter a single YouTube URL
       - Upload a text file with multiple YouTube URLs (one per line)
    
    2. **Adjust analysis options** if needed:
       - Frame interval: Determines how frequently frames are captured from the video
    
    3. **Click the Analyze button** to start processing
    
    4. **View and export results**:
       - Video summary (based on transcript if available)
       - Frame-by-frame analysis with visual captures and transcript segments
       - Download options for your analysis
    
    ### Notes
    - Processing may take several minutes depending on video length
    - For full transcript analysis, videos must have captions available
    - For best results, use videos under 20 minutes
    """)

# Footer
st.markdown("---")
st.markdown("Powered by Streamlit | YouTube Analyzer")