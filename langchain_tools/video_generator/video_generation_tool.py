"""
Video Generation Tool for LangChain Integration
Creates educational videos about Indian Constitution topics with Google TTS audio
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import uuid
import tempfile
import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Video generation dependencies (no MoviePy)
import requests
from urllib.parse import quote
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

class VideoGenerationInput(BaseModel):
    """Input schema for Video Generation Tool"""
    topic: str = Field(..., description="The constitutional topic to create a video about")
    duration_minutes: Optional[float] = Field(default=2.5, description="Target video duration in minutes (default: 2.5)")
    video_style: Optional[str] = Field(default="educational", description="Style: educational, animated, or presentation")
    difficulty: Optional[str] = Field(default="medium", description="Difficulty level: beginner, intermediate, or advanced")
    include_examples: Optional[bool] = Field(default=True, description="Whether to include practical examples")
    rag_context: Optional[str] = Field(default="", description="RAG context for constitutional information")

class VideoGenerationTool:
    """
    A tool for generating educational videos about constitutional topics.
    
    Features:
    - LLM-powered script generation
    - Google TTS for natural audio narration
    - Constitutional law focus with examples
    - Multiple visual styles
    - Customizable duration and difficulty
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.output_dir = "/home/shruthi/SME-Indian-Constitution-And-Rights/generated_videos/"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize LLM for script generation
        if self.gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.gemini_api_key,
                temperature=0.3  # Balanced creativity for educational content
            )
        else:
            self.llm = None
            print("Warning: GEMINI_API_KEY not found. Video generation will be limited.")
    
    def generate_video(self, 
                      topic: str, 
                      duration_minutes: float = 2.5, 
                      video_style: str = "educational",
                      difficulty: str = "medium",
                      include_examples: bool = True,
                      rag_context: str = "") -> Dict[str, Any]:
        """
        Main method to generate educational videos
        
        Args:
            topic: Constitutional topic to cover
            duration_minutes: Target video duration in minutes
            video_style: Style of video (educational, animated, presentation)
            difficulty: Difficulty level
            include_examples: Whether to include examples
            rag_context: RAG context for constitutional information
        
        Returns:
            Dictionary with video generation results
        """
        try:
            print(f"🎬 Starting video generation for: {topic}")
            start_time = datetime.now()
            
            # Step 1: Generate script content
            script_data = self._generate_script_with_llm(
                topic, duration_minutes, difficulty, include_examples, rag_context
            )
            
            if not script_data.get('success'):
                return script_data
            
            # Step 2: Generate audio using Google TTS
            audio_file = self._generate_audio_with_google_tts(
                script_data['script_text'], 
                script_data['segments']
            )
            print(f"Audio files: {audio_file}")
            # Step 3: Create visual content
            visual_clips = self._create_visual_content(
                script_data['segments'], 
                video_style, 
                duration_minutes
            )
            
            # Step 4: Combine audio and visuals
            video_file = self._create_final_video(
                audio_file, 
                visual_clips, 
                topic, 
                script_data
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Video generation completed in {processing_time:.1f} seconds")
            print(f"📹 Video saved at: {video_file}")
            
            return {
                'success': True,
                'video_path': video_file,
                'audio_path': audio_file,
                'script_data': script_data,
                'topic': topic,
                'duration': duration_minutes,
                'processing_time': processing_time,
                'created_at': end_time.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error in video generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'topic': topic
            }
    
    def _generate_script_with_llm(self, topic: str, duration: float, difficulty: str, include_examples: bool, rag_context: str) -> Dict[str, Any]:
        """Generate video script using LLM"""
        
        if not self.llm:
            return {'success': False, 'error': 'LLM not available'}
        
        # Calculate target word count (average speaking pace: 150-180 WPM)
        target_words = int(duration * 160)  # 160 WPM average
        
        prompt = f"""
You are an expert constitutional law educator creating a video script about the Indian Constitution.

Topic: {topic}
Target Duration: {duration} minutes ({target_words} words approximately)
Difficulty Level: {difficulty}
Include Examples: {include_examples}

RAG Context (Use this authoritative information):
{rag_context}

Please create an engaging, educational video script with the following structure:

1. **INTRODUCTION** (15% of content)
   - Hook the audience with an interesting fact or question
   - Briefly introduce the topic and its importance

2. **MAIN CONTENT** (70% of content)
   - Break down the topic into 3-4 clear segments
   - Use simple, accessible language
   - Include specific constitutional articles, sections, or amendments
   - {"Add practical examples and real-world applications" if include_examples else "Focus on theoretical understanding"}
   - Connect concepts to everyday citizen rights and responsibilities

3. **CONCLUSION** (15% of content)
   - Summarize key points
   - End with actionable takeaways or reflection questions

FORMAT REQUIREMENTS:
- Write in a conversational, engaging tone
- Use short sentences and clear transitions
- Mark natural pauses with [PAUSE]
- Segment the script for visual slides with [SLIDE: Title]
- Target approximately {target_words} words total
- Focus specifically on Indian Constitutional law

EXAMPLE FORMAT:
[SLIDE: Introduction]
Did you know that Article 21 of the Indian Constitution... [PAUSE]

[SLIDE: What is Article 21?]
The right to life and personal liberty is fundamental... [PAUSE]

Please generate the complete script following this structure and focusing on {topic}.
"""

        try:
            response = self.llm.invoke(prompt)
            script_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse script into segments
            segments = self._parse_script_segments(script_text)
            print(f"Segments: {segments}")
            return {
                'success': True,
                'script_text': script_text,
                'segments': segments,
                'word_count': len(script_text.split()),
                'estimated_duration': len(script_text.split()) / 160  # 160 WPM
            }
            
        except Exception as e:
            print(f"❌ Error generating script: {e}")
            return {'success': False, 'error': f'Script generation failed: {str(e)}'}
    
    def _clean_script_text(self, raw_script: str) -> str:
        """Clean up the raw script text from LLM to remove formatting and unwanted content"""
        
        # Remove common LLM meta-responses
        meta_phrases = [
            "Of course, I can generate",
            "I'll create",
            "Here's a",
            "I'll help you create",
            "Let me create",
            "I can help",
            "Here is",
            "I'll generate",
            "Certainly!",
            "Absolutely!",
            "Sure!"
        ]
        
        cleaned_text = raw_script
        
        # Remove meta phrases at the beginning
        for phrase in meta_phrases:
            if cleaned_text.strip().lower().startswith(phrase.lower()):
                # Find the end of the sentence and remove it
                sentences = cleaned_text.split('.')
                if len(sentences) > 1:
                    cleaned_text = '.'.join(sentences[1:]).strip()
        
        # Remove markdown formatting
        import re
        
        # Remove bold (**text**)
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)
        
        # Remove italic (*text*)
        cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)
        
        # Remove headers (### text)
        cleaned_text = re.sub(r'^#{1,6}\s+(.*?)$', r'\1', cleaned_text, flags=re.MULTILINE)
        
        # Remove bullet points (- text)
        cleaned_text = re.sub(r'^[-*]\s+', '', cleaned_text, flags=re.MULTILINE)
        
        # Remove numbered lists (1. text)
        cleaned_text = re.sub(r'^\d+\.\s+', '', cleaned_text, flags=re.MULTILINE)
        
        # Remove extra whitespace and normalize
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Multiple newlines to double
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        
        # Remove common section headers that aren't slide markers
        section_headers = [
            "INTRODUCTION:",
            "MAIN CONTENT:",
            "CONCLUSION:",
            "FORMAT REQUIREMENTS:",
            "EXAMPLE FORMAT:"
        ]
        
        for header in section_headers:
            cleaned_text = cleaned_text.replace(header, '')
        
        # Clean up around slide markers
        lines = cleaned_text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Skip lines that are just formatting instructions
                if line.startswith('Please generate') or \
                   line.startswith('Target approximately') or \
                   line.startswith('Focus specifically') or \
                   'words total' in line.lower():
                    continue
                cleaned_lines.append(line)
        
        # Rejoin and final cleanup
        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = cleaned_text.strip()
        
        print(f"📝 Script cleaned. Original length: {len(raw_script)}, Cleaned length: {len(cleaned_text)}")
        
        return cleaned_text
    def _parse_script_segments(self, script_text: str) -> List[Dict[str, Any]]:
        """Parse script into segments for video creation"""
        segments = []
        current_slide = "Introduction"
        current_text = []
        
        lines = script_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('[SLIDE:'):
                # Save previous segment
                if current_text:
                    segments.append({
                        'slide_title': current_slide,
                        'text': ' '.join(current_text).replace('[PAUSE]', ''),
                        'duration': len(' '.join(current_text).split()) / 160 * 60  # seconds
                    })
                    current_text = []
                
                # Extract new slide title
                current_slide = line.replace('[SLIDE:', '').replace(']', '').strip()
            elif line and not line.startswith('['):
                current_text.append(line)
        
        # Add final segment
        if current_text:
            segments.append({
                'slide_title': current_slide,
                'text': ' '.join(current_text).replace('[PAUSE]', ''),
                'duration': len(' '.join(current_text).split()) / 160 * 60
            })
        
        return segments
    
    def _clean_segment_text(self, text_lines: List[str]) -> str:
        """Clean segment text content to remove unwanted phrases and formatting"""
        import re
        
        # Join the text lines
        raw_text = ' '.join(text_lines)
        
        # Remove [PAUSE] markers
        cleaned_text = raw_text.replace('[PAUSE]', '')
        
        # Remove common LLM meta-phrases that might appear in segments
        meta_phrases_to_remove = [
            r"^(Of course,?\s*)",
            r"^(Here'?s\s+)",
            r"^(Here\s+is\s+)",
            r"^(I'?ll\s+create\s+)",
            r"^(I'?ll\s+help\s+you\s+create\s+)",
            r"^(Let me\s+create\s+)",
            r"^(I\s+can\s+help\s+)",
            r"^(I'?ll\s+generate\s+)",
            r"^(Certainly!?\s*)",
            r"^(Absolutely!?\s*)",
            r"^(Sure!?\s*)",
            r"^(Now,?\s+let'?s\s+)",
            r"^(Let'?s\s+)",
            r"^(We\s+can\s+)"
        ]
        
        for pattern in meta_phrases_to_remove:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove any remaining markdown formatting
        cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned_text)  # Bold
        cleaned_text = re.sub(r'\*(.*?)\*', r'\1', cleaned_text)      # Italic
        cleaned_text = re.sub(r'#{1,6}\s*', '', cleaned_text)         # Headers
        
        # Remove bullet points and numbering that might leak through
        cleaned_text = re.sub(r'^[-*•]\s+', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'^\d+\.\s+', '', cleaned_text, flags=re.MULTILINE)
        
        # Remove instructional phrases that might appear in content
        instructional_phrases = [
            r"Please note that",
            r"As we can see",
            r"It'?s important to understand",
            r"Let'?s explore",
            r"We need to understand",
            r"It'?s worth noting",
            r"Keep in mind",
            r"Remember that",
            r"Don'?t forget"
        ]
        
        for phrase in instructional_phrases:
            cleaned_text = re.sub(phrase, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        cleaned_text = cleaned_text.strip()
        
        # Remove empty sentences and clean punctuation
        sentences = cleaned_text.split('.')
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep substantial sentences
                # Ensure proper capitalization
                if sentence and not sentence[0].isupper():
                    sentence = sentence[0].upper() + sentence[1:]
                clean_sentences.append(sentence)
        
        # Rejoin sentences
        if clean_sentences:
            cleaned_text = '. '.join(clean_sentences)
            if not cleaned_text.endswith('.'):
                cleaned_text += '.'
        else:
            cleaned_text = ""
        
        print(f"🧹 Segment cleaned: '{raw_text[:50]}...' -> '{cleaned_text[:50]}...'")
        
        return cleaned_text
    
    def _generate_audio_with_google_tts(self, script_text: str, segments: List[Dict]) -> str:
        """Generate audio using Google Text-to-Speech API"""
        
        # Clean script text for TTS (remove slide markers and pauses)
        clean_text = script_text.replace('[PAUSE]', '. ')
        clean_text = ' '.join([line for line in clean_text.split('\n') 
                              if not line.strip().startswith('[SLIDE:')])
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"constitution_video_audio_{timestamp}.mp3"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        try:
            # Use Google TTS API (via gTTS library or direct API call)
            # For now, using gTTS as it's simpler to implement
            from gtts import gTTS
            
            tts = gTTS(
                text=clean_text,
                lang='en',
                slow=False,
                tld='co.in'  # Indian English accent
            )
            
            tts.save(audio_path)
            print(f"🔊 Audio generated: {audio_path}")
            
            return audio_path
            
        except ImportError:
            print("f❌ gTTS library not found. Installing...")
            # Fallback: Use system TTS or generate silent audio
            return self._create_silent_audio(len(clean_text.split()) / 160 * 60)
        except Exception as e:
            print(f"❌ TTS generation failed: {e}")
            # Create silent audio as fallback
            return self._create_silent_audio(len(clean_text.split()) / 160 * 60)
    
    def _create_silent_audio(self, duration_seconds: float) -> str:
        """Create silent audio file as fallback using FFmpeg"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"silent_audio_{timestamp}.wav"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        try:
            # Create silent audio using FFmpeg
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 
                f'anullsrc=channel_layout=stereo:sample_rate=44100',
                '-t', str(duration_seconds),
                '-c:a', 'pcm_s16le',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"🔇 Silent audio created: {audio_path}")
                return audio_path
            else:
                # Fallback: create a very basic WAV file manually
                return self._create_basic_wav_silence(duration_seconds)
                
        except Exception as e:
            print(f"❌ Silent audio creation failed: {e}")
            # Fallback: create a very basic WAV file manually
            return self._create_basic_wav_silence(duration_seconds)
    
    def _create_basic_wav_silence(self, duration_seconds: float) -> str:
        """Create a basic silent WAV file manually"""
        import wave
        import struct
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"silent_audio_{timestamp}.wav"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        # WAV file parameters
        sample_rate = 44100
        num_channels = 2
        sample_width = 2
        num_frames = int(sample_rate * duration_seconds)
        
        with wave.open(audio_path, 'w') as wav_file:
            wav_file.setparams((num_channels, sample_width, sample_rate, num_frames, 'NONE', 'not compressed'))
            # Write silence (zeros)
            silence_data = struct.pack('<' + 'h' * (num_frames * num_channels), 
                                     *([0] * num_frames * num_channels))
            wav_file.writeframes(silence_data)
        
        print(f"🔇 Basic silent audio created: {audio_path}")
        return audio_path
    
    def _create_visual_content(self, segments: List[Dict], style: str, duration: float) -> List[str]:
        """Create visual content for each segment"""
        slide_paths = []
        
        for i, segment in enumerate(segments):
            # Create slide image
            slide_path = self._create_slide_image(
                segment['slide_title'], 
                segment['text'], 
                i + 1, 
                len(segments)
            )
            slide_paths.append({
                'image_path': slide_path,
                'duration': segment.get('duration', 10)
            })
        
        return slide_paths
    
    def _create_slide_image(self, title: str, text: str, slide_num: int, total_slides: int) -> str:
        """Create a slide image with title and content"""
        
        # Image dimensions (16:9 aspect ratio)
        width, height = 1920, 1080
        
        # Create blank image
        img = Image.new('RGB', (width, height), color='#1a237e')  # Constitutional blue
        draw = ImageDraw.Draw(img)
        
        try:
            # Load fonts (fallback to default if not available)
            title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 72)
            text_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
            footer_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
        except:
            # Fallback to default fonts
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default() 
            footer_font = ImageFont.load_default()
        
        # Draw title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            ((width - title_width) // 2, 100), 
            title, 
            fill='#ffeb3b',  # Golden yellow
            font=title_font
        )
        
        # Draw content text (wrapped)
        wrapped_text = textwrap.fill(text, width=80)
        text_lines = wrapped_text.split('\n')
        
        y_offset = 300
        line_height = 60
        
        for line in text_lines[:12]:  # Limit to 12 lines
            if y_offset + line_height > height - 200:  # Leave space for footer
                break
            
            line_bbox = draw.textbbox((0, 0), line, font=text_font)
            line_width = line_bbox[2] - line_bbox[0]
            draw.text(
                ((width - line_width) // 2, y_offset), 
                line, 
                fill='white',
                font=text_font
            )
            y_offset += line_height
        
        # Draw footer
        footer_text = f"Indian Constitution & Rights | Slide {slide_num}/{total_slides}"
        footer_bbox = draw.textbbox((0, 0), footer_text, font=footer_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        draw.text(
            ((width - footer_width) // 2, height - 80), 
            footer_text, 
            fill='#ffeb3b',
            font=footer_font
        )
        
        # Save slide image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slide_filename = f"slide_{slide_num}_{timestamp}.png"
        slide_path = os.path.join(self.output_dir, slide_filename)
        
        img.save(slide_path)
        print(f"🖼️ Slide created: {slide_path}")
        
        return slide_path
    
    def _create_final_video(self, audio_path: str, visual_clips: List[Dict], topic: str, script_data: Dict) -> str:
        """Combine audio and visuals into final video using FFmpeg"""
        
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')[:50]  # Limit length
            
            video_filename = f"constitutional_video_{safe_topic}_{timestamp}.mp4"
            video_path = os.path.join(self.output_dir, video_filename)
            
            # Create a temporary file list for FFmpeg concat
            concat_file = os.path.join(self.output_dir, f"concat_list_{timestamp}.txt")
            
            # Build concat file content
            concat_content = []
            for clip in visual_clips:
                image_path = clip['image_path']
                duration = clip['duration']
                concat_content.append(f"file '{image_path}'")
                concat_content.append(f"duration {duration}")
            
            # Add the last image again to ensure proper ending
            if visual_clips:
                concat_content.append(f"file '{visual_clips[-1]['image_path']}'")
            
            # Write concat file
            with open(concat_file, 'w') as f:
                f.write('\n'.join(concat_content))
            
            print(f"📄 Created concat file: {concat_file}")
            
            # Create video from images using FFmpeg
            temp_video = os.path.join(self.output_dir, f"temp_video_{timestamp}.mp4")
            
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
                '-c:v', 'libx264',
                '-r', '24',
                '-pix_fmt', 'yuv420p',
                temp_video
            ]
            
            print(f"🎬 Creating video from slides...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ FFmpeg video creation failed: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            # Combine with audio
            final_cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', audio_path,
                '-map', '0:v',      # ALWAYS take video from file 0
                '-map', '1:a',      # ALWAYS take audio from file 1
                '-c:v', 'libx264',  # DO NOT COPY video stream
                '-c:a', 'aac',      
                '-b:a', '128k',
                '-shortest',
                video_path
            ]

            print(f"🔊 Muxing video + audio: {" ".join(final_cmd)}")

            result = subprocess.run(final_cmd, capture_output=True, text=True)
            print(f"Return code: {result.returncode}")
            print(f"FFmpeg stderr: {result.stderr}")

            if result.returncode != 0:
                print(f"❌ FFmpeg audio merge failed: {result.stderr}")
                # If audio merge fails, at least return the video without audio
                if os.path.exists(temp_video):
                    os.rename(temp_video, video_path)
                    print(f"⚠️ Video created without audio: {video_path}")
                else:
                    raise Exception(f"FFmpeg audio merge failed: {result.stderr}")
            
            # Clean up temporary files
            try:
                os.remove(concat_file)
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                # Clean up slide images
                for clip in visual_clips:
                    if os.path.exists(clip['image_path']):
                        os.remove(clip['image_path'])
                print("🧹 Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️ Could not clean up some temporary files: {e}")
            
            return video_path
            
        except Exception as e:
            print(f"❌ Error creating final video: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create a simple slideshow without audio
            return self._create_fallback_slideshow(visual_clips, topic, timestamp)
    
    def _create_fallback_slideshow(self, visual_clips: List[Dict], topic: str, timestamp: str) -> str:
        """Create a simple slideshow as fallback when FFmpeg fails"""
        
        try:
            # Generate output filename for fallback
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')[:50]
            
            fallback_video = os.path.join(self.output_dir, f"slideshow_{safe_topic}_{timestamp}.mp4")
            
            if not visual_clips:
                print("❌ No visual clips available for fallback")
                return ""
            
            # Use the first image and create a simple video
            first_image = visual_clips[0]['image_path']
            
            simple_cmd = [
                'ffmpeg', '-y',
                '-loop', '1',
                '-i', first_image,
                '-t', '30',  # 30 seconds fallback
                '-c:v', 'libx264',
                '-r', '24',
                '-pix_fmt', 'yuv420p',
                fallback_video
            ]
            
            result = subprocess.run(simple_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"📹 Fallback slideshow created: {fallback_video}")
                return fallback_video
            else:
                print(f"❌ Fallback slideshow creation failed: {result.stderr}")
                return ""
                
        except Exception as e:
            print(f"❌ Error in fallback slideshow: {e}")
            return ""


# Create the tool function for LangChain integration
@tool("video_generation_tool", args_schema=VideoGenerationInput)
def video_generation_tool(
    topic: str,
    duration_minutes: float = 2.5,
    video_style: str = "educational", 
    difficulty: str = "medium",
    include_examples: bool = True,
    rag_context: str = ""
) -> str:
    """
    Generate educational videos about Indian Constitution topics.
    
    This tool creates comprehensive video content including:
    - LLM-generated scripts focused on constitutional law
    - Google TTS audio narration  
    - Visual slides with constitutional content
    - Professional video output suitable for education
    
    Args:
        topic: The constitutional topic (e.g., "Fundamental Rights", "Article 21", "Directive Principles")
        duration_minutes: Target video length in minutes (default: 2.5)
        video_style: Visual style - educational, animated, or presentation (default: educational) 
        difficulty: Content difficulty - beginner, intermediate, or advanced (default: medium)
        include_examples: Whether to include practical examples and case studies (default: True)
        rag_context: Constitutional knowledge context from RAG system
    
    Returns:
        JSON string with video generation results including file path and metadata
    """
    
    try:
        print(f"🎬 Video Generation Tool called with topic: {topic}")
        generator = VideoGenerationTool()
        result = generator.generate_video(
            topic=topic,
            duration_minutes=duration_minutes,
            video_style=video_style,
            difficulty=difficulty, 
            include_examples=include_examples,
            rag_context=rag_context
        )
        
        print(f"📊 Video generation result: {result.get('success', False)}")
        
        if result['success']:
            output_message = f"""✅ Constitutional video generated successfully!

📹 Video Details:
- Topic: {topic}
- Duration: {duration_minutes} minutes
- Video File: {result['video_path']}
- Audio File: {result['audio_path']}
- Processing Time: {result.get('processing_time', 0):.1f} seconds

📝 Script Summary:
- Word Count: {result['script_data'].get('word_count', 'N/A')}
- Estimated Duration: {result['script_data'].get('estimated_duration', 0):.1f} minutes
- Segments: {len(result['script_data'].get('segments', []))}

Files saved:
- Video: {result['video_path']}
- Audio: {result['audio_path']}

The video and audio files are ready for use in constitutional education!"""
            print(f"✅ Returning success message to LangChain orchestrator")
            return output_message
        else:
            error_message = f"❌ Video generation failed: {result.get('error', 'Unknown error')}"
            print(f"❌ Returning error message to LangChain orchestrator")
            return error_message
            
    except Exception as e:
        error_message = f"❌ Error in video generation tool: {str(e)}"
        print(f"❌ Exception in video generation tool: {e}")
        import traceback
        traceback.print_exc()
        return error_message


# Export the tool instance
video_generation_tool_instance = VideoGenerationTool()
