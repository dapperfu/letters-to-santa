#!/usr/bin/env python3
"""
Voice extraction module for Letters to Santa videos.

This module extracts audio from videos and performs speaker diarization
using pyannote.audio, organizing voices by speaker into separate folders.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json

# Try to import pyannote.audio (optional dependency)
try:
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio not available. Voice extraction will be skipped.")


def extract_audio_from_video(video_path: str, output_audio_path: str) -> None:
    """
    Extract audio track from video using ffmpeg.
    
    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_audio_path : str
        Path to save extracted audio file (WAV format).
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If audio extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to extract audio.")
    
    # Extract audio using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit
        '-ar', '16000',  # Sample rate 16kHz (common for speech)
        '-ac', '1',  # Mono
        '-y',  # Overwrite output file
        output_audio_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")


def perform_speaker_diarization(
    audio_path: str,
    huggingface_token: Optional[str] = None
) -> List[Dict]:
    """
    Perform speaker diarization on audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file.
    huggingface_token : Optional[str]
        Hugging Face access token for pyannote models.
        If None, tries to use default or environment variable.
    
    Returns
    -------
    List[Dict]
        List of speaker segments with start, end, and speaker_id.
    
    Raises
    ------
    RuntimeError
        If diarization fails or pyannote.audio is not available.
    """
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError("pyannote.audio is not available. Install it to use voice extraction.")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get Hugging Face token from environment if not provided
    if huggingface_token is None:
        huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not huggingface_token:
        print("Warning: No Hugging Face token provided. Some models may require authentication.")
        print("Set HUGGINGFACE_TOKEN environment variable or pass --token argument.")
    
    try:
        # Load the pre-trained speaker diarization pipeline
        if huggingface_token:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=huggingface_token
            )
        else:
            # Try without token (may work for some models)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
    except Exception as e:
        # Fallback to community model
        try:
            if huggingface_token:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    use_auth_token=huggingface_token
                )
            else:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1"
                )
        except Exception as e2:
            raise RuntimeError(f"Failed to load diarization pipeline: {e2}")
    
    # Move pipeline to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print(f"Performing speaker diarization on {audio_path}...")
    print(f"Using device: {device}")
    
    # Apply the pipeline
    with ProgressHook() as hook:
        diarization = pipeline(audio_path, hook=hook)
    
    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': float(turn.start),
            'end': float(turn.end),
            'speaker_id': speaker
        })
    
    return segments


def extract_voice_segments(
    audio_path: str,
    segments: List[Dict],
    output_dir: str,
    video_name: str
) -> None:
    """
    Extract and save voice segments per speaker.
    
    Parameters
    ----------
    audio_path : str
        Path to audio file.
    segments : List[Dict]
        List of speaker segments from diarization.
    output_dir : str
        Base output directory for voice folders.
    video_name : str
        Base name of the video.
    
    Raises
    ------
    OSError
        If directory creation fails.
    RuntimeError
        If segment extraction fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group segments by speaker
    speakers: Dict[str, List[Dict]] = {}
    for segment in segments:
        speaker_id = segment['speaker_id']
        if speaker_id not in speakers:
            speakers[speaker_id] = []
        speakers[speaker_id].append(segment)
    
    print(f"Found {len(speakers)} unique speaker(s)")
    
    # Extract segments for each speaker
    for speaker_id, speaker_segments in speakers.items():
        # Create folder for this speaker
        speaker_folder = output_path / f"speaker_{speaker_id}"
        speaker_folder.mkdir(exist_ok=True)
        
        # Save all segments for this speaker
        segment_files = []
        for i, segment in enumerate(speaker_segments):
            start_time = segment['start']
            end_time = segment['end']
            duration = end_time - start_time
            
            # Extract segment using ffmpeg
            segment_filename = f"segment_{i+1:03d}.wav"
            segment_path = speaker_folder / segment_filename
            
            cmd = [
                'ffmpeg',
                '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'copy',
                '-y',
                str(segment_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segment_files.append({
                    'filename': segment_filename,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to extract segment {i+1} for {speaker_id}: {e}")
        
        # Save metadata
        metadata = {
            'speaker_id': speaker_id,
            'video_name': video_name,
            'total_segments': len(speaker_segments),
            'segments': segment_files
        }
        
        metadata_file = speaker_folder / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Speaker {speaker_id}: Extracted {len(segment_files)} segments")


def extract_voices_from_video(
    video_path: str,
    output_dir: str = 'voices',
    huggingface_token: Optional[str] = None
) -> None:
    """
    Extract voices from a video file.
    
    Parameters
    ----------
    video_path : str
        Path to video file.
    output_dir : str
        Directory to save voice segments (default: 'voices').
    huggingface_token : Optional[str]
        Hugging Face access token for pyannote models.
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If extraction fails or pyannote.audio is not available.
    """
    if not PYANNOTE_AVAILABLE:
        print("Skipping voice extraction: pyannote.audio is not available.")
        print("Install pyannote.audio to enable voice extraction:")
        print("  pip install pyannote.audio")
        return
    
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    print(f"\n{'='*60}")
    print(f"Extracting voices from: {video_name}")
    print(f"{'='*60}")
    
    # Create temporary audio file
    temp_audio_dir = Path(output_dir) / "temp"
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_audio_dir / f"{video_name}.wav"
    
    try:
        # Extract audio from video
        print("Extracting audio from video...")
        extract_audio_from_video(video_path, str(temp_audio_path))
        
        # Perform speaker diarization
        segments = perform_speaker_diarization(str(temp_audio_path), huggingface_token)
        
        if not segments:
            print("No speaker segments detected in audio.")
            return
        
        # Extract voice segments
        extract_voice_segments(str(temp_audio_path), segments, output_dir, video_name)
        
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()
    
    print(f"Voice extraction complete for {video_name}")


def process_video(
    video_path: str,
    output_dir: str = 'voices',
    huggingface_token: Optional[str] = None
) -> None:
    """
    Process a single video for voice extraction.
    
    Parameters
    ----------
    video_path : str
        Path to video file.
    output_dir : str
        Directory to save voice segments (default: 'voices').
    huggingface_token : Optional[str]
        Hugging Face access token for pyannote models.
    """
    extract_voices_from_video(video_path, output_dir, huggingface_token)


def process_all_videos(
    videos_dir: str = 'videos',
    output_dir: str = 'voices',
    huggingface_token: Optional[str] = None
) -> None:
    """
    Process all videos in a directory.
    
    Parameters
    ----------
    videos_dir : str
        Directory containing video files (default: 'videos').
    output_dir : str
        Directory to save voice segments (default: 'voices').
    huggingface_token : Optional[str]
        Hugging Face access token for pyannote models.
    
    Raises
    ------
    FileNotFoundError
        If videos directory does not exist.
    """
    if not PYANNOTE_AVAILABLE:
        print("Skipping voice extraction: pyannote.audio is not available.")
        return
    
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    # Find all video files
    video_extensions = {'.webm', '.mp4', '.avi', '.mov', '.mkv'}
    video_files = [
        f for f in videos_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]
    
    if not video_files:
        print(f"No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video file(s) to process")
    print("")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        try:
            process_video(str(video_file), output_dir, huggingface_token)
            successful += 1
            print("")
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            failed += 1
            print("")
    
    print(f"{'='*60}")
    print(f"Voice extraction complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract voices from videos using speaker diarization'
    )
    parser.add_argument(
        '--videos-dir',
        default='videos',
        help='Directory containing video files (default: videos)'
    )
    parser.add_argument(
        '--output-dir',
        default='voices',
        help='Directory to save voice segments (default: voices)'
    )
    parser.add_argument(
        '--file',
        help='Process a single video file instead of all in directory'
    )
    parser.add_argument(
        '--token',
        help='Hugging Face access token for pyannote models'
    )
    
    args = parser.parse_args()
    
    # Gracefully handle missing pyannote.audio (optional dependency)
    if not PYANNOTE_AVAILABLE:
        print("Warning: pyannote.audio is not available. Voice extraction will be skipped.", file=sys.stderr)
        print("Install it with: pip install pyannote.audio", file=sys.stderr)
        return
    
    try:
        if args.file:
            process_video(args.file, args.output_dir, args.token)
        else:
            process_all_videos(args.videos_dir, args.output_dir, args.token)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
