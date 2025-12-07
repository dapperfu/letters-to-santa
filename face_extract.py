#!/usr/bin/env python3
"""
Face extraction and clustering module for Letters to Santa videos.

This module extracts faces from video key frames using InsightFace,
clusters similar faces together, and organizes them into separate
folders with multiple samples per person.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import insightface
import json


def extract_faces_from_video(
    video_path: str,
    keyframe_interval: float = 1.0
) -> List[Dict]:
    """
    Extract faces from key frames of a video.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    keyframe_interval : float
        Interval in seconds between key frames (default: 1.0).
    
    Returns
    -------
    List[Dict]
        List of face data dictionaries containing:
        - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
        - 'embedding': Face encoding/embedding (512-dimensional)
        - 'landmarks': Facial landmarks (eyes, nose, mouth)
        - 'frame_number': Frame number in video
        - 'timestamp': Timestamp in seconds
        - 'image': Cropped face image (numpy array)
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If face extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize InsightFace model
    app = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * keyframe_interval)
    
    faces_data: List[Dict] = []
    frame_number = 0
    
    print(f"Processing video: {video_path}")
    print(f"Extracting faces from key frames (every {keyframe_interval}s)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process key frames only
        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps
            
            # Detect faces
            faces = app.get(frame)
            
            for face in faces:
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Extract embedding (512-dimensional)
                embedding = face.embedding if hasattr(face, 'embedding') else face.normed_embedding
                
                # Extract landmarks (key points)
                landmarks = face.kps.astype(int) if hasattr(face, 'kps') else np.array([])
                
                # Crop face image
                face_image = frame[y1:y2, x1:x2]
                
                faces_data.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'embedding': embedding,
                    'landmarks': landmarks.tolist() if landmarks.size > 0 else [],
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'image': face_image
                })
        
        frame_number += 1
    
    cap.release()
    print(f"Extracted {len(faces_data)} faces from {frame_number} frames")
    
    return faces_data


def cluster_faces(
    face_encodings: np.ndarray,
    threshold: float = 0.6
) -> Dict[int, List[int]]:
    """
    Cluster similar faces using DBSCAN with cosine similarity.
    
    Parameters
    ----------
    face_encodings : np.ndarray
        Array of face encodings (N x 512).
    threshold : float
        Similarity threshold for clustering (default: 0.6).
        Lower values create tighter clusters.
    
    Returns
    -------
    Dict[int, List[int]]
        Mapping of cluster_id -> list of face indices.
        Cluster ID -1 indicates noise/outliers.
    """
    if len(face_encodings) == 0:
        return {}
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(face_encodings)
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Use DBSCAN with cosine distance
    # eps: maximum distance between samples in the same cluster
    # min_samples: minimum number of samples in a cluster
    clustering = DBSCAN(eps=1.0 - threshold, min_samples=1, metric='precomputed')
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Organize clusters
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)
    
    return clusters


def save_clustered_faces(
    faces: List[Dict],
    clusters: Dict[int, List[int]],
    output_dir: str,
    video_name: str,
    samples_per_person: int = 5
) -> None:
    """
    Save clustered faces into separate folders.
    
    Parameters
    ----------
    faces : List[Dict]
        List of face data dictionaries from extract_faces_from_video.
    clusters : Dict[int, List[int]]
        Cluster mapping from cluster_faces.
    output_dir : str
        Base output directory for face folders.
    video_name : str
        Base name of the video (for metadata).
    samples_per_person : int
        Number of random samples to save per person (default: 5).
    
    Raises
    ------
    OSError
        If directory creation fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter out noise cluster (-1) if present
    valid_clusters = {k: v for k, v in clusters.items() if k != -1}
    
    if not valid_clusters:
        print("No valid clusters found. Saving all faces as noise.")
        valid_clusters = {-1: clusters.get(-1, [])}
    
    print(f"Saving {len(valid_clusters)} unique face clusters...")
    
    for cluster_id, face_indices in valid_clusters.items():
        # Create folder for this person
        person_folder = output_path / f"person_{cluster_id:03d}"
        person_folder.mkdir(exist_ok=True)
        
        # Select random samples (up to samples_per_person)
        sample_indices = random.sample(
            face_indices,
            min(samples_per_person, len(face_indices))
        )
        
        # Save face images and metadata
        metadata = {
            'cluster_id': int(cluster_id),
            'total_faces': len(face_indices),
            'saved_samples': len(sample_indices),
            'video_name': video_name
        }
        
        for i, face_idx in enumerate(sample_indices):
            face_data = faces[face_idx]
            
            # Save face image
            face_image = face_data['image']
            if face_image.size > 0:
                filename = f"face_{i+1:03d}.jpg"
                filepath = person_folder / filename
                cv2.imwrite(str(filepath), face_image)
                
                # Save metadata for each face
                face_metadata = {
                    'filename': filename,
                    'bbox': face_data['bbox'],
                    'frame_number': face_data['frame_number'],
                    'timestamp': face_data['timestamp'],
                    'landmarks': face_data['landmarks']
                }
                
                metadata_file = person_folder / f"face_{i+1:03d}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(face_metadata, f, indent=2)
        
        # Save cluster metadata
        cluster_metadata_file = person_folder / "cluster_metadata.json"
        with open(cluster_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Person {cluster_id:03d}: Saved {len(sample_indices)} samples from {len(face_indices)} total faces")


def save_face_data_for_annotation(
    faces: List[Dict],
    clusters: Dict[int, List[int]],
    video_name: str,
    output_dir: str
) -> str:
    """
    Save face data in a format suitable for video annotation.
    
    Parameters
    ----------
    faces : List[Dict]
        List of face data dictionaries.
    clusters : Dict[int, List[int]]
        Cluster mapping.
    video_name : str
        Base name of the video.
    output_dir : str
        Directory to save face data.
    
    Returns
    -------
    str
        Path to saved face data JSON file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create annotation data structure
    annotation_data = {
        'video_name': video_name,
        'faces': []
    }
    
    # Map face indices to cluster IDs
    face_to_cluster = {}
    for cluster_id, face_indices in clusters.items():
        for face_idx in face_indices:
            face_to_cluster[face_idx] = int(cluster_id)
    
    # Save face data (without images for JSON)
    for idx, face in enumerate(faces):
        face_entry = {
            'face_index': idx,
            'cluster_id': face_to_cluster.get(idx, -1),
            'bbox': face['bbox'],
            'landmarks': face['landmarks'],
            'frame_number': face['frame_number'],
            'timestamp': face['timestamp']
        }
        annotation_data['faces'].append(face_entry)
    
    # Save to JSON
    json_path = output_path / f"{video_name}_face_data.json"
    with open(json_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return str(json_path)


def process_video(
    video_path: str,
    output_dir: str = 'faces',
    keyframe_interval: float = 1.0,
    similarity_threshold: float = 0.6,
    samples_per_person: int = 5
) -> None:
    """
    Process a single video: extract, cluster, and save faces.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : str
        Directory to save clustered faces (default: 'faces').
    keyframe_interval : float
        Interval in seconds between key frames (default: 1.0).
    similarity_threshold : float
        Similarity threshold for clustering (default: 0.6).
    samples_per_person : int
        Number of random samples per person (default: 5).
    
    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    RuntimeError
        If processing fails.
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Extract faces
    faces = extract_faces_from_video(video_path, keyframe_interval)
    
    if not faces:
        print(f"No faces detected in {video_path}")
        return
    
    # Extract encodings
    encodings = np.array([face['embedding'] for face in faces])
    
    # Cluster faces
    print(f"Clustering {len(faces)} faces...")
    clusters = cluster_faces(encodings, similarity_threshold)
    
    # Save clustered faces
    save_clustered_faces(
        faces,
        clusters,
        output_dir,
        video_name,
        samples_per_person
    )
    
    # Save face data for annotation
    annotation_dir = Path(output_dir) / "annotation_data"
    annotation_path = save_face_data_for_annotation(
        faces,
        clusters,
        video_name,
        str(annotation_dir)
    )
    print(f"Face data saved for annotation: {annotation_path}")


def process_all_videos(
    videos_dir: str = 'videos',
    output_dir: str = 'faces',
    keyframe_interval: float = 1.0,
    similarity_threshold: float = 0.6,
    samples_per_person: int = 5
) -> None:
    """
    Process all videos in a directory.
    
    Parameters
    ----------
    videos_dir : str
        Directory containing video files (default: 'videos').
    output_dir : str
        Directory to save clustered faces (default: 'faces').
    keyframe_interval : float
        Interval in seconds between key frames (default: 1.0).
    similarity_threshold : float
        Similarity threshold for clustering (default: 0.6).
    samples_per_person : int
        Number of random samples per person (default: 5).
    
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
    print("")
    
    successful = 0
    failed = 0
    
    for video_file in video_files:
        try:
            process_video(
                str(video_file),
                output_dir,
                keyframe_interval,
                similarity_threshold,
                samples_per_person
            )
            successful += 1
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            failed += 1
            print("")
    
    print(f"\n{'='*60}")
    print(f"Processing complete: {successful} successful, {failed} failed")
    print(f"{'='*60}")


def main() -> None:
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract and cluster faces from videos using InsightFace'
    )
    parser.add_argument(
        '--videos-dir',
        default='videos',
        help='Directory containing video files (default: videos)'
    )
    parser.add_argument(
        '--output-dir',
        default='faces',
        help='Directory to save clustered faces (default: faces)'
    )
    parser.add_argument(
        '--file',
        help='Process a single video file instead of all in directory'
    )
    parser.add_argument(
        '--keyframe-interval',
        type=float,
        default=1.0,
        help='Interval in seconds between key frames (default: 1.0)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.6,
        help='Similarity threshold for clustering (default: 0.6)'
    )
    parser.add_argument(
        '--samples-per-person',
        type=int,
        default=5,
        help='Number of random samples per person (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.file:
            process_video(
                args.file,
                args.output_dir,
                args.keyframe_interval,
                args.similarity_threshold,
                args.samples_per_person
            )
        else:
            process_all_videos(
                args.videos_dir,
                args.output_dir,
                args.keyframe_interval,
                args.similarity_threshold,
                args.samples_per_person
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
