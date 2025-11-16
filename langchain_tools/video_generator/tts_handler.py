"""
TTS Handler for Sarvam API Integration
Handles text-to-speech conversion using Sarvam's API
"""

import os
import requests
import json
from typing import List, Dict, Any
import time
from pathlib import Path

class SarvamTTSHandler:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.base_url = "https://api.sarvam.ai/text-to-speech"
        self.voice_id = "meera"  # Default Hindi-English voice
        
        if not self.api_key:
            print("Warning: SARVAM_API_KEY not found. TTS functionality will be limited.")
    
    def text_to_speech(self, text: str, output_path: str, voice_id: str = None) -> bool:
        """
        Convert text to speech using Sarvam API
        
        Args:
            text: Text to convert to speech
            output_path: Path where audio file will be saved
            voice_id: Voice to use (default: meera)
        
        Returns:
            bool: Success status
        """
        if not self.api_key:
            # Fallback: create silent audio or skip
            print(f"No API key available. Skipping TTS for: {text[:50]}...")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": [text],
                "target_language_code": "hi-IN",  # Hindi-English mix
                "speaker": voice_id or self.voice_id,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 22050,
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            }
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Save audio content
                audio_data = response.content
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                print(f"✅ TTS generated: {output_path}")
                return True
            else:
                print(f"❌ TTS API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ TTS error: {str(e)}")
            return False
    
    def generate_script_audio(self, script_segments: List[Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Generate audio files for multiple script segments
        
        Args:
            script_segments: List of dicts with 'text', 'duration', 'slide_id'
            output_dir: Directory to save audio files
        
        Returns:
            List of generated audio file paths
        """
        audio_files = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, segment in enumerate(script_segments):
            audio_file = output_path / f"segment_{i+1}_{segment.get('slide_id', 'unknown')}.wav"
            
            success = self.text_to_speech(
                text=segment['text'],
                output_path=str(audio_file)
            )
            
            if success:
                audio_files.append(str(audio_file))
            else:
                # Create placeholder or skip
                print(f"Skipping audio for segment {i+1}")
        
        return audio_files
    
    def estimate_speech_duration(self, text: str, words_per_minute: int = 150) -> float:
        """
        Estimate speech duration based on word count
        
        Args:
            text: Text to analyze
            words_per_minute: Average speaking rate
        
        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        duration_minutes = word_count / words_per_minute
        return duration_minutes * 60  # Convert to seconds