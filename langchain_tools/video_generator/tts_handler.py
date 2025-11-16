"""
TTS Handler for Sarvam API Integration
Handles text-to-speech conversion using Sarvam's API
"""

import os
import requests
from typing import List, Dict, Any
from pathlib import Path

class SarvamTTSHandler:
    def __init__(self, audio_format: str = "mp3"):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.base_url = "https://api.sarvam.ai/text-to-speech"
        self.voice_id = "manisha"
        self.audio_format = audio_format.lower()
        
        # Supported formats and their extensions
        self.supported_formats = {
            'mp3': '.mp3',
            'wav': '.wav',
            'aac': '.aac',
            'm4a': '.m4a'
        }
        
        if self.audio_format not in self.supported_formats:
            print(f"Warning: Audio format '{audio_format}' not supported. Using MP3 instead.")
            self.audio_format = "mp3"
        
        if not self.api_key:
            print("Warning: SARVAM_API_KEY not found. TTS functionality will be limited.")
    
    def text_to_speech(self, text: str, output_path: str, voice_id: str = None) -> bool:
        """
        Convert text to speech using Sarvam API and save in specified format
        """
        if not self.api_key:
            print(f"No API key available. Skipping TTS for: {text[:50]}...")
            return False
        
        try:
            headers = {
                "api-subscription-key": self.api_key,  # REQUIRED
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": [text],
                "target_language_code": "hi-IN",
                "speaker": voice_id or self.voice_id,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v2"
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Sarvam API returns WAV audio, we may need to convert
                temp_wav_path = None
                
                if self.audio_format == "wav":
                    # Direct save for WAV
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                else:
                    # Save as temporary WAV first, then convert
                    import tempfile
                    temp_wav_path = tempfile.mktemp(suffix=".wav")
                    
                    with open(temp_wav_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Convert to desired format
                    conversion_success = self._convert_audio_format(temp_wav_path, output_path)
                    
                    # Clean up temp file
                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
                    
                    if not conversion_success:
                        print(f"⚠️ Audio conversion failed, keeping as WAV")
                        # Fallback: save as WAV with original extension
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                
                print(f"✅ TTS generated: {output_path} (format: {self.audio_format})")
                return True
            
            print(f"❌ TTS API error: {response.status_code} - {response.text}")
            return False
                
        except Exception as e:
            print(f"❌ TTS error: {str(e)}")
            return False
    
    def _convert_audio_format(self, input_path: str, output_path: str) -> bool:
        """
        Convert audio from WAV to the specified format
        """
        try:
            # Try using pydub for audio conversion
            try:
                from pydub import AudioSegment
                
                # Load WAV file
                audio = AudioSegment.from_wav(input_path)
                
                # Export in desired format
                if self.audio_format == "mp3":
                    audio.export(output_path, format="mp3", bitrate="128k")
                elif self.audio_format == "aac":
                    audio.export(output_path, format="aac", bitrate="128k")
                elif self.audio_format == "m4a":
                    audio.export(output_path, format="mp4", bitrate="128k")
                else:
                    # Fallback to MP3
                    audio.export(output_path, format="mp3", bitrate="128k")
                
                print(f"✅ Audio converted to {self.audio_format}")
                return True
                
            except ImportError:
                print("⚠️ pydub not available for audio conversion")
                
                # Fallback: try using ffmpeg directly if available
                try:
                    import subprocess
                    
                    cmd = [
                        'ffmpeg', '-i', input_path, '-c:a',
                        'mp3' if self.audio_format == 'mp3' else 'aac',
                        '-b:a', '128k', '-y', output_path
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print(f"✅ Audio converted using ffmpeg to {self.audio_format}")
                        return True
                    else:
                        print(f"⚠️ ffmpeg conversion failed: {result.stderr}")
                        
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    print("⚠️ ffmpeg not available for audio conversion")
                
                return False
                
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return False
    
    def generate_script_audio(self, script_segments: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Generate audio files for multiple script segments in the specified format
        """
        audio_files = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, segment in enumerate(script_segments):
            # Use the configured audio format extension
            file_extension = self.supported_formats[self.audio_format]
            audio_file = output_path / f"segment_{i+1}_{segment.get('slide_id', 'unknown')}{file_extension}"
            print(f"Generating audio segment {i+1}/{len(script_segments)}: {audio_file.name}")
            
            success = self.text_to_speech(
                text=segment['content'],
                output_path=str(audio_file)
            )
            
            if success:
                audio_files.append(str(audio_file))
            else:
                print(f"Skipping audio for segment {i+1}")
        
        return audio_files
    
    def estimate_speech_duration(self, text: str, words_per_minute: int = 150) -> float:
        """
        Estimate speech duration based on word count
        """
        word_count = len(text.split())
        duration_minutes = word_count / words_per_minute
        return duration_minutes * 60
