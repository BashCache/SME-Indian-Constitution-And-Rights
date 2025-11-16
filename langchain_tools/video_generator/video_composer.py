"""
Video Composer
Handles video assembly using MoviePy, combining slides and audio
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
from datetime import datetime

try:
    from moviepy.editor import (
        ImageClip, AudioFileClip, CompositeVideoClip, 
        concatenate_videoclips, ColorClip, TextClip, VideoFileClip
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Video composition will be limited.")

class VideoComposer:
    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "constitutional_videos"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Video settings
        self.video_settings = {
            'fps': 24,
            'resolution': (1920, 1080),
            'codec': 'libx264',
            'audio_codec': 'aac',
            'bitrate': '2000k'
        }
    
    def create_video_from_slides(self, 
                                slides: List[str], 
                                audio_files: List[str], 
                                script_data: Dict[str, Any],
                                output_path: str,
                                target_duration: float = 150.0) -> bool:  # 2.5 minutes
        """
        Create video by combining slides and audio
        
        Args:
            slides: List of slide image paths
            audio_files: List of audio file paths
            script_data: Script timing and content information
            output_path: Where to save final video
            target_duration: Target video duration in seconds
        
        Returns:
            Success status
        """
        if not MOVIEPY_AVAILABLE:
            print("❌ MoviePy not available. Cannot create video.")
            return False
        
        try:
            video_clips = []
            total_duration = 0
            
            # Get segment durations from script data
            segments = script_data.get('segments', [])
            
            # Calculate duration per slide
            num_slides = len(slides)
            if num_slides == 0:
                print("❌ No slides provided for video creation")
                return False
            
            # Distribute time across slides based on content
            slide_durations = self._calculate_slide_durations(segments, target_duration)
            
            print(f"🎬 Creating video with {num_slides} slides, target duration: {target_duration}s")
            
            # Create clips for each slide
            for i, (slide_path, duration) in enumerate(zip(slides, slide_durations)):
                print(f"Processing slide {i+1}/{num_slides}: {duration:.1f}s")
                
                # Create image clip
                if os.path.exists(slide_path):
                    img_clip = ImageClip(slide_path, duration=duration)
                else:
                    # Create placeholder clip if slide image doesn't exist
                    img_clip = ColorClip(
                        size=self.video_settings['resolution'], 
                        color=(240, 240, 240), 
                        duration=duration
                    )
                
                # Add audio if available
                if i < len(audio_files) and os.path.exists(audio_files[i]):
                    try:
                        audio_clip = AudioFileClip(audio_files[i])
                        # Adjust audio to match slide duration
                        if audio_clip.duration > duration:
                            audio_clip = audio_clip.subclip(0, duration)
                        elif audio_clip.duration < duration:
                            # Extend with silence or loop
                            audio_clip = audio_clip.set_duration(duration)
                        
                        img_clip = img_clip.set_audio(audio_clip)
                    except Exception as e:
                        print(f"Warning: Could not add audio for slide {i+1}: {e}")
                
                video_clips.append(img_clip)
                total_duration += duration
            
            if not video_clips:
                print("❌ No valid video clips created")
                return False
            
            # Add transitions
            final_clips = self._add_transitions(video_clips)
            
            # Concatenate all clips
            print("🔄 Concatenating video clips...")
            final_video = concatenate_videoclips(final_clips, method="compose")
            
            # Set final properties
            final_video = final_video.set_fps(self.video_settings['fps'])
            
            # Add background music if needed (optional)
            final_video = self._add_background_music(final_video)
            
            # Export video
            print(f"📹 Exporting video to: {output_path}")
            final_video.write_videofile(
                output_path,
                fps=self.video_settings['fps'],
                codec=self.video_settings['codec'],
                audio_codec=self.video_settings['audio_codec'],
                bitrate=self.video_settings['bitrate'],
                verbose=False,
                logger=None  # Suppress moviepy logs
            )
            
            # Clean up
            final_video.close()
            for clip in video_clips:
                clip.close()
            
            print(f"✅ Video created successfully: {output_path}")
            print(f"📊 Final duration: {total_duration:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating video: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_slide_durations(self, segments: List[Dict], target_duration: float) -> List[float]:
        """
        Calculate how long each slide should be displayed
        
        Args:
            segments: Script segments with content info
            target_duration: Total target duration
        
        Returns:
            List of durations for each slide
        """
        if not segments:
            # Default equal distribution
            num_slides = 5  # Assume 5 slides
            return [target_duration / num_slides] * num_slides
        
        durations = []
        total_estimated = 0
        
        # Calculate based on content length
        for segment in segments:
            content_length = len(segment.get('content', ''))
            # Estimate duration based on content (roughly 150 words per minute)
            estimated_duration = max(15.0, min(40.0, content_length / 10))  # 15-40 seconds
            durations.append(estimated_duration)
            total_estimated += estimated_duration
        
        # Normalize to target duration
        if total_estimated > 0:
            scale_factor = target_duration / total_estimated
            durations = [d * scale_factor for d in durations]
        
        return durations
    
    def _add_transitions(self, clips: List) -> List:
        """
        Add subtle transitions between clips
        """
        if len(clips) <= 1:
            return clips
        
        try:
            # Simple crossfade transitions
            transition_duration = 0.5  # Half second transitions
            final_clips = []
            
            for i, clip in enumerate(clips):
                if i == 0:
                    # First clip: fade in
                    clip = clip.fadein(transition_duration)
                elif i == len(clips) - 1:
                    # Last clip: fade out
                    clip = clip.fadeout(transition_duration)
                else:
                    # Middle clips: crossfade
                    clip = clip.fadein(transition_duration).fadeout(transition_duration)
                
                final_clips.append(clip)
            
            return final_clips
            
        except Exception as e:
            print(f"Warning: Could not add transitions: {e}")
            return clips
    
    def _add_background_music(self, video_clip) -> Any:
        """
        Add subtle background music (optional)
        """
        # For now, return as-is
        # Could add royalty-free background music here
        return video_clip
    
    def create_title_sequence(self, title: str, subtitle: str, duration: float = 5.0) -> Any:
        """
        Create animated title sequence
        """
        if not MOVIEPY_AVAILABLE:
            return None
        
        try:
            # Create background
            bg = ColorClip(
                size=self.video_settings['resolution'],
                color=(0, 51, 102),  # Navy blue
                duration=duration
            )
            
            # Create title text
            title_clip = TextClip(
                title,
                fontsize=80,
                color='white',
                font='Arial-Bold'
            ).set_position('center').set_duration(duration)
            
            # Create subtitle text
            subtitle_clip = TextClip(
                subtitle,
                fontsize=40,
                color='orange',
                font='Arial'
            ).set_position(('center', 'center')).set_duration(duration)
            
            # Adjust subtitle position
            subtitle_clip = subtitle_clip.set_position(('center', self.video_settings['resolution'][1] * 0.6))
            
            # Composite
            title_sequence = CompositeVideoClip([bg, title_clip, subtitle_clip])
            
            return title_sequence
            
        except Exception as e:
            print(f"Warning: Could not create title sequence: {e}")
            return None
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about created video
        """
        if not os.path.exists(video_path):
            return {}
        
        try:
            file_size = os.path.getsize(video_path)
            
            if MOVIEPY_AVAILABLE:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                fps = clip.fps
                resolution = clip.size
                clip.close()
            else:
                duration = 0
                fps = 24
                resolution = (1920, 1080)
            
            return {
                'file_path': video_path,
                # 'file_size_mb': round(file_size / (1024 * 1024), 2),
                'duration_seconds': round(duration, 2),
                'fps': fps,
                'resolution': f"{resolution[0]}x{resolution[1]}" if resolution else "1920x1080",
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Warning: Could not get video info: {e}")
            return {'file_path': video_path, 'error': str(e)}