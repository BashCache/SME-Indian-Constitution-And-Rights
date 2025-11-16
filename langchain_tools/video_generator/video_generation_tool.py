"""
Video Generation Tool for LangChain Integration
Generates educational videos about constitutional topics
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import tempfile
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our video generation components
from .tts_handler import SarvamTTSHandler
from .slide_template_manager import SlideTemplateManager
from .video_composer import VideoComposer

class VideoGenerationInput(BaseModel):
    """Input schema for Video Generation Tool"""
    topic: str = Field(..., description="The constitutional topic to create video about")
    duration: Optional[float] = Field(default=150.0, description="Target video duration in seconds (default: 2.5 minutes)")
    style: Optional[str] = Field(default="educational", description="Video style: educational, formal, or casual")
    include_examples: Optional[bool] = Field(default=True, description="Whether to include real-world examples")

class VideoGenerationTool:
    """
    A comprehensive tool for generating educational videos about constitutional topics.
    
    Features:
    - Script generation using LLM
    - Text-to-speech conversion
    - Slide creation with constitutional templates
    - Video composition and assembly
    """
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.output_dir = Path("generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.tts_handler = SarvamTTSHandler()
        self.slide_manager = SlideTemplateManager()
        self.video_composer = VideoComposer()
        
        # Initialize LLM for script generation
        if self.gemini_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=self.gemini_api_key,
                temperature=0.3
            )
        else:
            self.llm = None
            print("Warning: GEMINI_API_KEY not found. Script generation will be limited.")
    
    def generate_video(self, topic: str, duration: float = 150.0, style: str = "educational", include_examples: bool = True) -> Dict[str, Any]:
        """
        Main method to generate a complete video
        
        Args:
            topic: Constitutional topic to cover
            duration: Target duration in seconds
            style: Video style
            include_examples: Whether to include examples
        
        Returns:
            Dictionary with generation results
        """
        try:
            print(f"🎬 Starting video generation for: {topic}")
            start_time = datetime.now()
            
            # Step 1: Generate script
            print("📝 Generating script...")
            script_data = self._generate_script(topic, duration, style, include_examples)
            
            if not script_data or not script_data.get('segments'):
                return {
                    'success': False,
                    'error': 'Failed to generate script content',
                    'topic': topic
                }
            
            # Step 2: Create slides
            print("🖼️ Creating slides...")
            temp_dir = Path(tempfile.mkdtemp(prefix="video_gen_"))
            slides_dir = temp_dir / "slides"
            audio_dir = temp_dir / "audio"
            
            slides = self.slide_manager.generate_slides_from_script(
                script_data, str(slides_dir)
            )
            
            # Step 3: Generate audio
            print("🎤 Generating audio...")
            audio_files = self.tts_handler.generate_script_audio(
                script_data['segments'], str(audio_dir)
            )
            
            # Step 4: Compose video
            print("🎞️ Composing video...")
            video_filename = f"{self._sanitize_filename(topic)}_{uuid.uuid4().hex[:8]}.mp4"
            output_path = self.output_dir / video_filename
            
            success = self.video_composer.create_video_from_slides(
                slides=slides,
                audio_files=audio_files,
                script_data=script_data,
                output_path=str(output_path),
                target_duration=duration
            )
            
            if success:
                # Get video information
                video_info = self.video_composer.get_video_info(str(output_path))
                
                # Clean up temporary files
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                print(f"✅ Video generation completed in {processing_time:.1f} seconds")
                
                return {
                    'success': True,
                    'video_path': str(output_path),
                    'video_info': video_info,
                    'script_data': script_data,
                    'topic': topic,
                    'processing_time': processing_time,
                    'created_at': end_time.isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Video composition failed',
                    'topic': topic
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
    
    def _generate_script(self, topic: str, duration: float, style: str, include_examples: bool) -> Dict[str, Any]:
        """
        Generate structured script for the video
        """
        if not self.llm:
            return self._generate_fallback_script(topic, duration)
        
        try:
            # Create prompt for script generation
            prompt = f"""
Create a structured script for a {duration/60:.1f}-minute educational video about "{topic}" related to the Indian Constitution.

Requirements:
- Video style: {style}
- Include examples: {include_examples}
- Target duration: {duration} seconds
- Suitable for general audience
- Focus on constitutional law and rights

Structure the response as a JSON object with this format:
{{
    "title": "Video title",
    "description": "Brief description",
    "total_duration": {duration},
    "segments": [
        {{
            "slide_id": "intro",
            "type": "title",
            "title": "Video title",
            "content": "Introduction text for narration",
            "duration": 15,
            "key_points": ["point1", "point2"]
        }},
        {{
            "slide_id": "main1",
            "type": "content",
            "title": "Section title",
            "content": "Main content for narration",
            "duration": 30,
            "key_points": ["point1", "point2", "point3"]
        }},
        // ... more segments ...
        {{
            "slide_id": "conclusion",
            "type": "conclusion",
            "title": "Conclusion",
            "content": "Summary and closing remarks",
            "duration": 20,
            "key_points": ["summary points"]
        }}
    ]
}}

Ensure:
1. Total segment durations approximately equal target duration
2. Content is accurate and educational
3. Language is clear and suitable for narration
4. Include constitutional articles/sections where relevant
5. Balance theory with practical examples if requested

Respond with ONLY the JSON object, no additional text."""
            
            response = self.llm.invoke(prompt)
            script_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean and parse JSON
            script_text = script_text.strip()
            if script_text.startswith('```json'):
                script_text = script_text[7:-3].strip()
            elif script_text.startswith('```'):
                script_text = script_text[3:-3].strip()
            
            script_data = json.loads(script_text)
            
            # Validate and adjust durations
            script_data = self._validate_script_data(script_data, duration)
            
            print(f"✅ Generated script with {len(script_data.get('segments', []))} segments")
            return script_data
            
        except Exception as e:
            print(f"Error generating script with LLM: {e}")
            return self._generate_fallback_script(topic, duration)
    
    def _generate_fallback_script(self, topic: str, duration: float) -> Dict[str, Any]:
        """
        Generate a basic fallback script when LLM is not available
        """
        segments_count = 5
        segment_duration = duration / segments_count
        
        fallback_script = {
            "title": f"Understanding {topic}",
            "description": f"An educational video about {topic} in the Indian Constitution",
            "total_duration": duration,
            "segments": [
                {
                    "slide_id": "intro",
                    "type": "title",
                    "title": f"Understanding {topic}",
                    "content": f"Welcome to this educational video about {topic}. Today we will explore this important constitutional concept and its significance in Indian law.",
                    "duration": segment_duration,
                    "key_points": ["Introduction", "Overview"]
                },
                {
                    "slide_id": "definition",
                    "type": "content",
                    "title": "Definition and Scope",
                    "content": f"Let us begin by understanding what {topic} means in the context of the Indian Constitution and how it is defined in our legal framework.",
                    "duration": segment_duration,
                    "key_points": ["Definition", "Constitutional basis"]
                },
                {
                    "slide_id": "importance",
                    "type": "content",
                    "title": "Constitutional Significance",
                    "content": f"The importance of {topic} in the Indian Constitution cannot be overstated. It plays a crucial role in protecting citizens' rights and ensuring democratic governance.",
                    "duration": segment_duration,
                    "key_points": ["Significance", "Democratic principles"]
                },
                {
                    "slide_id": "examples",
                    "type": "content",
                    "title": "Practical Applications",
                    "content": f"Let us look at some real-world examples of how {topic} is applied in practice and its impact on citizens' daily lives.",
                    "duration": segment_duration,
                    "key_points": ["Real-world examples", "Practical impact"]
                },
                {
                    "slide_id": "conclusion",
                    "type": "conclusion",
                    "title": "Summary and Conclusion",
                    "content": f"In conclusion, {topic} is a fundamental aspect of our constitutional framework. Understanding it helps us appreciate our rights and responsibilities as citizens.",
                    "duration": segment_duration,
                    "key_points": ["Summary", "Key takeaways"]
                }
            ]
        }
        
        print("✅ Generated fallback script")
        return fallback_script
    
    def _validate_script_data(self, script_data: Dict[str, Any], target_duration: float) -> Dict[str, Any]:
        """
        Validate and adjust script data
        """
        segments = script_data.get('segments', [])
        if not segments:
            return script_data
        
        # Calculate total duration
        total_segment_duration = sum(seg.get('duration', 30) for seg in segments)
        
        # Adjust if significantly different from target
        if abs(total_segment_duration - target_duration) > 30:  # 30 second tolerance
            scale_factor = target_duration / total_segment_duration
            for segment in segments:
                segment['duration'] = segment.get('duration', 30) * scale_factor
        
        script_data['total_duration'] = target_duration
        return script_data
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Create a safe filename from topic string
        """
        import re
        # Remove or replace unsafe characters
        safe_name = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename)
        safe_name = re.sub(r'\s+', '_', safe_name.strip())
        return safe_name[:50]  # Limit length

# Create the LangChain tool instance
video_tool_instance = VideoGenerationTool()

@tool("video_generation_tool", args_schema=VideoGenerationInput, return_direct=True)
def video_generation_tool(topic: str, duration: float = 150.0, style: str = "educational", include_examples: bool = True) -> str:
    """
    Generate an educational video about constitutional topics.
    
    This tool creates a complete video with slides, narration, and proper formatting.
    Perfect for explaining constitutional concepts, fundamental rights, and legal principles.
    
    Args:
        topic: The constitutional topic to create video about
        duration: Target video duration in seconds (default: 2.5 minutes)
        style: Video style - educational, formal, or casual
        include_examples: Whether to include real-world examples
    
    Returns:
        JSON string with video generation results
    """
    result = video_tool_instance.generate_video(
        topic=topic,
        duration=duration,
        style=style,
        include_examples=include_examples
    )
    
    # Format response for user
    if result['success']:
        video_info = result.get('video_info', {})
        response = f"""🎬 **Video Generated Successfully!**

📹 **Video Details:**
• **Topic:** {result['topic']}
• **Duration:** {video_info.get('duration_seconds', duration)} seconds
• **File Size:** {video_info.get('file_size_mb', 'Unknown')} MB
• **Resolution:** {video_info.get('resolution', '1920x1080')}
• **File Path:** {result['video_path']}

⏱️ **Processing Time:** {result.get('processing_time', 0):.1f} seconds

📝 **Content Structure:**
{len(result.get('script_data', {}).get('segments', []))} segments covering:
"""
        
        # Add segment overview
        script_data = result.get('script_data', {})
        for i, segment in enumerate(script_data.get('segments', [])[:3], 1):
            response += f"\n{i}. {segment.get('title', 'Content')}"
        
        if len(script_data.get('segments', [])) > 3:
            response += f"\n... and {len(script_data.get('segments', [])) - 3} more segments"
        
        response += f"\n\n✅ **Your educational video about '{result['topic']}' is ready for use!**"
        
    else:
        response = f"❌ **Video Generation Failed**\n\n**Topic:** {result['topic']}\n**Error:** {result.get('error', 'Unknown error')}\n\nPlease try again or check the logs for more details."
    
    return response