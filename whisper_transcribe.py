#!/usr/bin/env python3
"""
Whisper transcription module for Letters to Santa videos.

This module transcribes video files using OpenAI Whisper, generating
both SRT subtitle files and plain text transcripts. Supports GPU
acceleration when available.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import whisper
import torch


def detect_device() -> str:
    """
    Detect available compute device (GPU or CPU).
    
    Returns
    -------
    str
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def transcribe_video(
    video_path: str,
    model_name: str = 'base',
    device: Optional[str] = None
) -> Tuple[str, list]:
    """
    Transcribe a video file using Whisper.
    
    Parameters
    ----------
    video_path : str
        Path to the video file to transcribe.
    model_name : str
        Whisper model to use (tiny, base, small, medium, large).
        Default is 'base'.
    device : Optional[str]
        Device to use ('cuda', 'mps', 'cpu'). If None, auto-detect.
    
    Returns
    -------
    Tuple[str, list]
        Tuple containing:
        - Full transcript text (str)
        - List of segments with timestamps (list of dicts)
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If transcription fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if device is None:
        device = detect_device()
    
    print(f"Loading Whisper model '{model_name}' on device '{device}'...")
    model = whisper.load_model(model_name, device=device)
    
    print(f"Transcribing {video_path}...")
    result = model.transcribe(video_path)
    
    full_text = result['text'].strip()
    segments = result.get('segments', [])
    
    return full_text, segments


def generate_srt(segments: list) -> str:
    """
    Generate SRT subtitle format from Whisper segments.
    
    Parameters
    ----------
    segments : list
        List of segment dictionaries from Whisper transcription.
    
    Returns
    -------
    str
        SRT formatted subtitle text.
    """
    srt_lines = []
    
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment['start'])
        end_time = format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
    
    return "\n".join(srt_lines)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds into SRT timestamp format (HH:MM:SS,mmm).
    
    Parameters
    ----------
    seconds : float
        Time in seconds.
    
    Returns
    -------
    str
        Formatted timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def process_video(
    video_path: str,
    output_dir: Optional[str] = None,
    model_name: str = 'base'
) -> Tuple[str, str]:
    """
    Process a video file and generate both SRT and text transcripts.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : Optional[str]
        Directory to save transcripts. If None, uses same directory as video.
    model_name : str
        Whisper model to use. Default is 'base'.
    
    Returns
    -------
    Tuple[str, str]
        Paths to generated SRT and text files.
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If transcription fails.
    """
    video_path_obj = Path(video_path)
    
    if output_dir is None:
        output_dir = video_path_obj.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = video_path_obj.stem
    
    # Transcribe video
    full_text, segments = transcribe_video(video_path, model_name)
    
    # Generate SRT file
    srt_content = generate_srt(segments)
    srt_path = output_dir / f"{base_name}.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    # Generate text file
    txt_path = output_dir / f"{base_name}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Generated transcripts:")
    print(f"  SRT: {srt_path}")
    print(f"  Text: {txt_path}")
    
    return str(srt_path), str(txt_path)


def process_all_videos(
    videos_dir: str = 'videos',
    model_name: str = 'base'
) -> None:
    """
    Process all video files in a directory.
    
    Parameters
    ----------
    videos_dir : str
        Directory containing video files.
    model_name : str
        Whisper model to use. Default is 'base'.
    
    Raises
    ------
    FileNotFoundError
        If videos directory does not exist.
    """
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
    print(f"Using device: {detect_device()}")
    print("")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        try:
            print(f"Processing: {video_file.name}")
            process_video(str(video_file), model_name=model_name)
            successful += 1
            print("")
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            failed += 1
            print("")
    
    print(f"Transcription complete: {successful} successful, {failed} failed")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Transcribe videos using OpenAI Whisper'
    )
    parser.add_argument(
        '--videos-dir',
        default='videos',
        help='Directory containing video files (default: videos)'
    )
    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model to use (default: base)'
    )
    parser.add_argument(
        '--file',
        help='Transcribe a single video file instead of all in directory'
    )
    
    args = parser.parse_args()
    
    try:
        if args.file:
            process_video(args.file, model_name=args.model)
        else:
            process_all_videos(args.videos_dir, model_name=args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

