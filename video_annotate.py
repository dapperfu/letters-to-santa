#!/usr/bin/env python3
"""
Video annotation module for Letters to Santa videos.

This module annotates videos with face detection results, drawing
bounding boxes, face IDs, and facial landmarks on detected faces.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2


def load_face_data(face_data_path: str) -> Dict:
    """
    Load face data from JSON file.
    
    Parameters
    ----------
    face_data_path : str
        Path to face data JSON file.
    
    Returns
    -------
    Dict
        Face data dictionary containing video name and face information.
    
    Raises
    ------
    FileNotFoundError
        If face data file does not exist.
    """
    if not os.path.exists(face_data_path):
        raise FileNotFoundError(f"Face data file not found: {face_data_path}")
    
    with open(face_data_path, 'r') as f:
        return json.load(f)


def get_cluster_color(cluster_id: int) -> tuple:
    """
    Get a distinct color for a cluster ID.
    
    Parameters
    ----------
    cluster_id : int
        Cluster/person ID.
    
    Returns
    -------
    tuple
        BGR color tuple for OpenCV.
    """
    # Generate distinct colors using HSV color space
    hue = (cluster_id * 137) % 180  # Golden angle approximation
    color_hsv = np.uint8([[[hue, 255, 255]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, color_bgr))


def draw_landmarks(
    frame: np.ndarray,
    landmarks: List[List[int]],
    color: tuple
) -> None:
    """
    Draw facial landmarks on frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Video frame to draw on.
    landmarks : List[List[int]]
        List of landmark coordinates [[x, y], ...].
    color : tuple
        BGR color for landmarks.
    """
    if not landmarks:
        return
    
    landmarks_array = np.array(landmarks, dtype=np.int32)
    
    # InsightFace typically provides 5 key points:
    # 0: left eye, 1: right eye, 2: nose, 3: left mouth corner, 4: right mouth corner
    if len(landmarks_array) == 5:
        # Draw 5-point landmarks
        # Left eye
        if len(landmarks_array) > 0:
            cv2.circle(frame, tuple(landmarks_array[0]), 3, color, -1)
        # Right eye
        if len(landmarks_array) > 1:
            cv2.circle(frame, tuple(landmarks_array[1]), 3, color, -1)
        # Nose
        if len(landmarks_array) > 2:
            cv2.circle(frame, tuple(landmarks_array[2]), 3, color, -1)
        # Mouth corners
        if len(landmarks_array) > 3:
            cv2.circle(frame, tuple(landmarks_array[3]), 3, color, -1)
        if len(landmarks_array) > 4:
            cv2.circle(frame, tuple(landmarks_array[4]), 3, color, -1)
        
        # Draw connections
        # Eyes
        if len(landmarks_array) >= 2:
            cv2.line(frame, tuple(landmarks_array[0]), tuple(landmarks_array[1]), color, 1)
        # Nose to eye center
        if len(landmarks_array) >= 3:
            eye_center = ((landmarks_array[0] + landmarks_array[1]) // 2).astype(int)
            cv2.line(frame, tuple(eye_center), tuple(landmarks_array[2]), color, 1)
        # Mouth
        if len(landmarks_array) >= 5:
            cv2.line(frame, tuple(landmarks_array[3]), tuple(landmarks_array[4]), color, 1)
    else:
        # For other landmark formats, draw all points
        for landmark in landmarks_array:
            cv2.circle(frame, tuple(landmark), 2, color, -1)
        
        # If we have 106-point landmarks, draw connections
        if len(landmarks_array) >= 106:
            # InsightFace 106-point model
            left_eye_indices = [36, 37, 38, 39, 40, 41]  # Left eye contour
            right_eye_indices = [42, 43, 44, 45, 46, 47]  # Right eye contour
            nose_indices = [27, 28, 29, 30, 31, 32, 33, 34, 35]  # Nose
            mouth_indices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]  # Mouth
            
            # Draw eye contours
            for indices in [left_eye_indices, right_eye_indices]:
                if max(indices) < len(landmarks_array):
                    eye_points = landmarks_array[indices]
                    for i in range(len(eye_points) - 1):
                        pt1 = tuple(eye_points[i])
                        pt2 = tuple(eye_points[i + 1])
                        cv2.line(frame, pt1, pt2, color, 1)
                    # Close the contour
                    if len(eye_points) > 0:
                        cv2.line(frame, tuple(eye_points[-1]), tuple(eye_points[0]), color, 1)
            
            # Draw nose
            if max(nose_indices) < len(landmarks_array):
                nose_points = landmarks_array[nose_indices]
                for i in range(len(nose_points) - 1):
                    pt1 = tuple(nose_points[i])
                    pt2 = tuple(nose_points[i + 1])
                    cv2.line(frame, pt1, pt2, color, 1)
            
            # Draw mouth
            if max(mouth_indices) < len(landmarks_array):
                mouth_points = landmarks_array[mouth_indices]
                for i in range(len(mouth_points) - 1):
                    pt1 = tuple(mouth_points[i])
                    pt2 = tuple(mouth_points[i + 1])
                    cv2.line(frame, pt1, pt2, color, 1)
                # Close the contour
                if len(mouth_points) > 0:
                    cv2.line(frame, tuple(mouth_points[-1]), tuple(mouth_points[0]), color, 1)


def annotate_video(
    video_path: str,
    face_data_path: str,
    output_path: str
) -> None:
    """
    Annotate a video with face detection results.
    
    Parameters
    ----------
    video_path : str
        Path to input video file.
    face_data_path : str
        Path to face data JSON file.
    output_path : str
        Path to save annotated video.
    
    Raises
    ------
    FileNotFoundError
        If video or face data file does not exist.
    RuntimeError
        If video processing fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Load face data
    face_data = load_face_data(face_data_path)
    
    # Organize faces by frame number
    faces_by_frame: Dict[int, List[Dict]] = {}
    for face in face_data['faces']:
        frame_num = face['frame_number']
        if frame_num not in faces_by_frame:
            faces_by_frame[frame_num] = []
        faces_by_frame[frame_num].append(face)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_path}")
    
    print(f"Annotating video: {video_path}")
    print(f"Output: {output_path}")
    
    frame_number = 0
    faces_drawn = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get faces for this frame
        frame_faces = faces_by_frame.get(frame_number, [])
        
        # Draw annotations for each face
        for face in frame_faces:
            bbox = face['bbox']
            cluster_id = face['cluster_id']
            landmarks = face['landmarks']
            
            # Get color for this cluster
            color = get_cluster_color(cluster_id)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw face ID label
            label = f"Person {cluster_id:03d}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(
                frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 5),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw landmarks
            if landmarks:
                draw_landmarks(frame, landmarks, color)
            
            faces_drawn += 1
        
        # Write frame
        out.write(frame)
        frame_number += 1
        
        if frame_number % 30 == 0:
            print(f"  Processed {frame_number} frames, drawn {faces_drawn} face annotations")
    
    cap.release()
    out.release()
    
    print(f"Annotation complete: {frame_number} frames, {faces_drawn} face annotations")


def process_video(
    video_path: str,
    face_data_path: Optional[str] = None,
    output_dir: str = 'videos_annotated'
) -> None:
    """
    Process a single video for annotation.
    
    Parameters
    ----------
    video_path : str
        Path to video file.
    face_data_path : Optional[str]
        Path to face data JSON. If None, searches in faces/annotation_data.
    output_dir : str
        Directory to save annotated video (default: 'videos_annotated').
    
    Raises
    ------
    FileNotFoundError
        If video or face data file does not exist.
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # Find face data file if not provided
    if face_data_path is None:
        annotation_data_dir = Path('faces') / 'annotation_data'
        face_data_path = annotation_data_dir / f"{video_name}_face_data.json"
        
        if not face_data_path.exists():
            raise FileNotFoundError(
                f"Face data not found for {video_name}. "
                f"Run face extraction first: make extract-faces"
            )
    
    # Create output path
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    output_path = output_path_obj / f"{video_name}_annotated.mp4"
    
    annotate_video(str(video_path), str(face_data_path), str(output_path))


def process_all_videos(
    videos_dir: str = 'videos',
    output_dir: str = 'videos_annotated'
) -> None:
    """
    Process all videos in a directory.
    
    Parameters
    ----------
    videos_dir : str
        Directory containing video files (default: 'videos').
    output_dir : str
        Directory to save annotated videos (default: 'videos_annotated').
    
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
    
    print(f"Found {len(video_files)} video file(s) to annotate")
    print("")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        try:
            process_video(str(video_file), output_dir=output_dir)
            successful += 1
            print("")
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            failed += 1
            print("")
    
    print(f"{'='*60}")
    print(f"Annotation complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Annotate videos with face detection results'
    )
    parser.add_argument(
        '--videos-dir',
        default='videos',
        help='Directory containing video files (default: videos)'
    )
    parser.add_argument(
        '--output-dir',
        default='videos_annotated',
        help='Directory to save annotated videos (default: videos_annotated)'
    )
    parser.add_argument(
        '--file',
        help='Annotate a single video file instead of all in directory'
    )
    parser.add_argument(
        '--face-data',
        help='Path to face data JSON file (auto-detected if not provided)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.file:
            process_video(args.file, args.face_data, args.output_dir)
        else:
            process_all_videos(args.videos_dir, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
