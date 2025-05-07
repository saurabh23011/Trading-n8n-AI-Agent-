#test
import sys
import argparse
import json
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, NoTranscriptAvailable
from youtube_transcript_api._transcripts import Transcript
import http.client
import requests


class YouTubeTranscriptApp:
    def __init__(self):
        self.parser = self._create_parser()
        
    def _create_parser(self):
        """Create argument parser for command line interface"""
        parser = argparse.ArgumentParser(
            description='Fetch and process YouTube video transcripts'
        )
        
        parser.add_argument(
            'video_id',
            help='YouTube video ID (e.g., the "dQw4w9WgXcQ" part from https://www.youtube.com/watch?v=dQw4w9WgXcQ)'
        )
        
        parser.add_argument(
            '--languages', '-l',
            nargs='+',
            default=['en'],
            help='Preferred language(s) for transcript (e.g., "en" "fr"). Default is "en"'
        )
        
        parser.add_argument(
            '--output', '-o',
            help='Save transcript to specified file'
        )
        
        parser.add_argument(
            '--format', '-f',
            choices=['text', 'json', 'srt'],
            default='text',
            help='Output format (text, json, or srt). Default is text'
        )
        
        parser.add_argument(
            '--list-languages', '-ll',
            action='store_true',
            help='List available transcript languages for the video'
        )
        
        parser.add_argument(
            '--search', '-s',
            help='Search for a specific word or phrase in the transcript'
        )
        
        return parser
    
    def get_transcript(self, video_id, languages, proxies=None, cookies=None):
        """Fetch transcript for the given video ID and languages"""
        try:
            if proxies or cookies:
                # Use proxies or cookies if provided
                return YouTubeTranscriptApi.get_transcript(
                    video_id, 
                    languages=languages,
                    proxies=proxies,
                    cookies=cookies
                )
            else:
                return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        except NoTranscriptFound:
            print(f"Error: No transcript found for video {video_id} in specified languages {languages}")
            return None
        except TranscriptsDisabled:
            print(f"Error: Transcripts are disabled for video {video_id}")
            return None
        except VideoUnavailable:
            print(f"Error: Video {video_id} is unavailable")
            return None
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return None
    
    def list_available_languages(self, video_id, proxies=None):
        """List all available transcript languages for a video"""
        try:
            if proxies:
                # Custom implementation using proxies
                from youtube_transcript_api._api import TranscriptListFetcher
                fetcher = TranscriptListFetcher()
                transcript_list = fetcher.fetch(video_id, proxies=proxies)
            else:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
            print(f"Available languages for video {video_id}:")
            for transcript in transcript_list:
                print(f"- {transcript.language_code}: {transcript.language}")
            return transcript_list
        except Exception as e:
            print(f"Error retrieving language list: {e}")
            return None
    
    def format_transcript_as_text(self, transcript):
        """Format transcript as plain text"""
        return "\n".join([entry['text'] for entry in transcript])
    
    def format_transcript_as_json(self, transcript):
        """Format transcript as JSON"""
        return json.dumps(transcript, indent=2)
    
    def format_transcript_as_srt(self, transcript):
        """Format transcript as SRT subtitle format"""
        srt_content = ""
        for i, entry in enumerate(transcript):
            start_time = entry['start']
            # Calculate end time (use next segment start or add duration)
            if i < len(transcript) - 1:
                end_time = transcript[i+1]['start']
            else:
                end_time = start_time + entry.get('duration', 5)
                
            # Format times as HH:MM:SS,mmm
            start_formatted = self._format_srt_time(start_time)
            end_formatted = self._format_srt_time(end_time)
            
            srt_content += f"{i+1}\n"
            srt_content += f"{start_formatted} --> {end_formatted}\n"
            srt_content += f"{entry['text']}\n\n"
            
        return srt_content
    
    def _format_srt_time(self, seconds):
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds_remainder = seconds % 60
        milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"
    
    def search_transcript(self, transcript, search_term):
        """Search for a term in the transcript and return matching entries with timestamps"""
        search_term = search_term.lower()
        results = []
        
        for entry in transcript:
            if search_term in entry['text'].lower():
                results.append({
                    'time': self._format_time(entry['start']),
                    'text': entry['text']
                })
        
        return results
    
    def _format_time(self, seconds):
        """Format seconds to MM:SS format"""
        minutes = int(seconds / 60)
        seconds_remainder = int(seconds % 60)
        return f"{minutes:02d}:{seconds_remainder:02d}"
    
    def save_to_file(self, content, filename):
        """Save content to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Transcript saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    def run(self, args=None):
        """Run the application with the provided arguments"""
        if args is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(args)
        
        video_id = args.video_id
        
        # List available languages if requested
        if args.list_languages:
            return self.list_available_languages(video_id)
        
        # Get the transcript
        transcript = self.get_transcript(video_id, args.languages)
        if not transcript:
            return False
        
        # Search if requested
        if args.search:
            results = self.search_transcript(transcript, args.search)
            if results:
                print(f"Found {len(results)} matches for '{args.search}':")
                for result in results:
                    print(f"[{result['time']}] {result['text']}")
            else:
                print(f"No matches found for '{args.search}'")
            return True
        
        # Format transcript according to requested format
        if args.format == 'text':
            formatted_transcript = self.format_transcript_as_text(transcript)
        elif args.format == 'json':
            formatted_transcript = self.format_transcript_as_json(transcript)
        elif args.format == 'srt':
            formatted_transcript = self.format_transcript_as_srt(transcript)
        
        # Save to file or print to console
        if args.output:
            return self.save_to_file(formatted_transcript, args.output)
        else:
            print(formatted_transcript)
            return True


def main():
    """Main entry point"""
    app = YouTubeTranscriptApp()
    success = app.run()
    return 0 if success else 1


def extract_video_id(url):
    """
    Extract the YouTube video ID from a URL
    
    Handles different YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    """
    # Regular expressions for different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # For watch?v= and youtu.be/ formats
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # For embed/ format
        r'^([0-9A-Za-z_-]{11})$'             # Direct video ID
    ]
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def test_proxy(proxy):
    """Test if a proxy is working"""
    try:
        response = requests.get("https://www.youtube.com", proxies={"http": proxy, "https": proxy}, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def streamlit_app():
    """
    Streamlit web interface for the YouTube Transcript application
    """
    st.set_page_config(
        page_title="YouTube Transcript Tool",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("YouTube Transcript Tool")
    st.markdown("""
    This tool allows you to fetch and analyze transcripts from YouTube videos.
    Enter a YouTube video URL or ID to get started.
    """)
    
    # Initialize session state if not exists
    if 'proxy_url' not in st.session_state:
        st.session_state.proxy_url = ""
    
    # Add proxy support information
    with st.expander("⚠️ IP Block Issues and Proxy Support"):
        st.markdown("""
        ### YouTube IP Block Issues
        
        YouTube may block transcript requests from certain IPs, especially:
        - After making too many requests in a short time
        - When using cloud provider IPs (AWS, Google Cloud, Azure, etc.)
        
        ### Proxy Solutions
        
        To bypass IP blocks, you can use a proxy server. Enter a proxy URL in the format:
        ```
        http://username:password@host:port
        ```
        or
        ```
        http://host:port
        ```
        
        **Note**: Free proxies are often unreliable. For serious use, consider a paid proxy service.
        """)
        
        # Add proxy input field
        proxy_url = st.text_input("Proxy URL (optional):", placeholder="http://username:password@host:port")
        
        # Store proxy URL in session state
        if proxy_url:
            st.session_state.proxy_url = proxy_url
        
        if proxy_url:
            if st.button("Test Proxy"):
                with st.spinner("Testing proxy connection..."):
                    if test_proxy(proxy_url):
                        st.success("✅ Proxy is working!")
                    else:
                        st.error("❌ Proxy connection failed. Please check the URL or try another proxy.")
    
    st.markdown("---")
    
    # Create an instance of the transcript app
    app = YouTubeTranscriptApp()
    
    # Get video URL or ID from user
    video_input = st.text_input("Enter YouTube video URL or ID:", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ or dQw4w9WgXcQ")
    
    # Process only if input is provided
    if video_input:
        video_id = extract_video_id(video_input)
        
        if not video_id:
            st.error("Invalid YouTube URL or ID. Please enter a valid YouTube video URL or ID.")
            st.stop()
        
        # Display video embedding
        st.video(f"https://www.youtube.com/watch?v={video_id}")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["Transcript", "Search", "Available Languages"])
        
        with tab1:
            st.header("Transcript")
            
            # Language selection
            available_languages = [("en", "English (en)")]  # Default option
            
            try:
                use_proxy_for_langs = st.checkbox("Use Proxy for Language List", key="proxy_for_langs")
                proxies = None
                if use_proxy_for_langs and st.session_state.proxy_url:
                    proxies = {
                        "http": st.session_state.proxy_url,
                        "https": st.session_state.proxy_url
                    }
                
                transcript_list = app.list_available_languages(video_id, proxies=proxies)
                if transcript_list:
                    available_languages = [(t.language_code, f"{t.language} ({t.language_code})") for t in transcript_list]
            except Exception as e:
                st.warning(f"Could not retrieve language list: {e}. Using English as default.")
            
            selected_language = st.selectbox(
                "Select language:",
                options=[lang_code for lang_code, _ in available_languages],
                format_func=lambda x: next((name for code, name in available_languages if code == x), x)
            )
            
            # Format selection
            format_options = {
                "text": "Plain Text",
                "json": "JSON",
                "srt": "SRT Subtitles"
            }
            selected_format = st.selectbox(
                "Select output format:",
                options=list(format_options.keys()),
                format_func=lambda x: format_options[x]
            )
            
            # Get transcript button
            col1, col2 = st.columns(2)
            with col1:
                get_transcript_btn = st.button("Get Transcript")
            with col2:
                use_proxy = st.checkbox("Use Proxy", help="Enable to use the proxy defined in the expander above")
                
            if get_transcript_btn:
                with st.spinner("Fetching transcript..."):
                    try:
                        proxies = None
                        if use_proxy and st.session_state.proxy_url:
                            proxies = {
                                "http": st.session_state.proxy_url,
                                "https": st.session_state.proxy_url
                            }
                            st.info("Using proxy to fetch transcript...")
                        
                        transcript = app.get_transcript(video_id, [selected_language], proxies=proxies)
                        
                        if transcript:
                            # Format according to selection
                            if selected_format == "text":
                                formatted_transcript = app.format_transcript_as_text(transcript)
                                st.text_area("Transcript:", formatted_transcript, height=400)
                            elif selected_format == "json":
                                formatted_transcript = app.format_transcript_as_json(transcript)
                                st.json(transcript)
                            elif selected_format == "srt":
                                formatted_transcript = app.format_transcript_as_srt(transcript)
                                st.text_area("SRT Transcript:", formatted_transcript, height=400)
                            
                            # Download button
                            file_extension = "txt" if selected_format == "text" else selected_format
                            st.download_button(
                                label="Download Transcript",
                                data=formatted_transcript,
                                file_name=f"transcript_{video_id}.{file_extension}",
                                mime=f"text/{file_extension}" if selected_format != "json" else "application/json"
                            )
                        else:
                            st.error("Failed to retrieve transcript. Try using a proxy if you're experiencing IP blocks.")
                    except Exception as e:
                        st.error(f"Error fetching transcript: {e}")
                        st.info("If you're seeing an IP block error, try using a proxy in the expander section above.")
        
        with tab2:
            st.header("Search Transcript")
            search_term = st.text_input("Enter search term:")
            
            # Only use available languages if we successfully fetched them
            search_language = st.selectbox(
                "Select language for search:",
                options=[lang_code for lang_code, _ in available_languages],
                format_func=lambda x: next((name for code, name in available_languages if code == x), x),
                key="search_language"
            )
            
            search_use_proxy = st.checkbox("Use Proxy for Search", 
                                          help="Enable to use the proxy defined in the expander above",
                                          key="search_use_proxy")
            
            if search_term and st.button("Search", key="search_button"):
                with st.spinner("Searching transcript..."):
                    try:
                        proxies = None
                        if search_use_proxy and st.session_state.proxy_url:
                            proxies = {
                                "http": st.session_state.proxy_url,
                                "https": st.session_state.proxy_url
                            }
                            st.info("Using proxy to fetch transcript...")
                        
                        transcript = app.get_transcript(video_id, [search_language], proxies=proxies)
                        if transcript:
                            results = app.search_transcript(transcript, search_term)
                            if results:
                                st.success(f"Found {len(results)} matches")
                                for i, result in enumerate(results):
                                    with st.expander(f"Match {i+1} at {result['time']}"):
                                        st.write(result['text'])
                                        st.link_button(
                                            "Go to timestamp",
                                            f"https://www.youtube.com/watch?v={video_id}&t={int(int(result['time'].split(':')[0])*60) + int(result['time'].split(':')[1])}"
                                        )
                            else:
                                st.info(f"No matches found for '{search_term}'")
                        else:
                            st.error("Failed to retrieve transcript. Try using a proxy if you're experiencing IP blocks.")
                    except Exception as e:
                        st.error(f"Error searching transcript: {e}")
                        st.info("If you're seeing an IP block error, try using a proxy in the expander section above.")
        
        with tab3:
            st.header("Available Languages")
            
            lang_use_proxy = st.checkbox("Use Proxy for Language List", 
                                        help="Enable to use the proxy defined in the expander above",
                                        key="lang_use_proxy")
            
            if st.button("List Available Languages", key="list_lang_button"):
                with st.spinner("Fetching language list..."):
                    try:
                        proxies = None
                        if lang_use_proxy and st.session_state.proxy_url:
                            proxies = {
                                "http": st.session_state.proxy_url,
                                "https": st.session_state.proxy_url
                            }
                            st.info("Using proxy to fetch language list...")
                        
                        transcript_list = app.list_available_languages(video_id, proxies=proxies)
                            
                        if transcript_list:
                            st.subheader(f"Available languages for video {video_id}:")
                            
                            # Create a dataframe for better display
                            languages_data = []
                            for transcript in transcript_list:
                                languages_data.append({
                                    "Code": transcript.language_code,
                                    "Language": transcript.language,
                                    "Is Generated": "Yes" if transcript.is_generated else "No",
                                    "Is Default": "Yes" if transcript.is_default else "No"
                                })
                            
                            st.dataframe(languages_data)
                        else:
                            st.error("Failed to retrieve language list. Try using a proxy if you're experiencing IP blocks.")
                    except Exception as e:
                        st.error(f"Error retrieving language list: {e}")
                        st.info("If you're seeing an IP block error, try using a proxy in the expander section above.")
    
    # Alternative methods section
    st.markdown("---")
    with st.expander("🛠️ Alternative Methods to Get Transcripts"):
        st.markdown("""
        ### Other Ways to Get YouTube Transcripts
        
        If you're consistently facing IP blocks, here are some alternative approaches:
        
        1. **Use YouTube's Official Interface**:
           - Go to the video on YouTube
           - Click the "..." button below the video
           - Select "Show transcript"
           - Copy and paste the transcript manually
        
        2. **Use yt-dlp Command Line Tool**:
           ```bash
           pip install yt-dlp
           yt-dlp --write-auto-sub --skip-download "https://www.youtube.com/watch?v=VIDEO_ID"
           ```
        
        3. **Try Browser Extensions**:
           - Several browser extensions can extract YouTube transcripts directly
        
        4. **Use a Residential Proxy Service**:
           - Services like Smartproxy, Bright Data, or Oxylabs provide residential IPs that are less likely to be blocked
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)")
    st.caption("⚠️ Note: This tool relies on the YouTube Transcript API, which may be affected by YouTube's IP blocking policies. Use responsibly.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run as CLI application if --cli flag is provided
        sys.argv.pop(1)  # Remove --cli flag
        sys.exit(main())
    else:
        # Run as Streamlit application by default
        streamlit_app()