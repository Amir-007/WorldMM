#!/usr/bin/env python3
"""
Extract visual embeddings from video files.
Reads video paths from JSON files and generates embeddings using EmbeddingModel.
Supports split processing across multiple GPUs and automatic merging.
"""

import json
import pickle
import numpy as np
import os
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

from worldmm.embedding.embedding_wrapper import EmbeddingModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_video_entries(json_path: str) -> List[dict]:
    """Load caption entries from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_video_paths(json_path: str) -> List[str]:
    """Load video paths from JSON file."""
    data = load_video_entries(json_path)
    video_paths = [item['video_path'] for item in data if 'video_path' in item]
    print(f"Loaded {len(video_paths)} video paths from {json_path}")
    return video_paths


def process_videos_sequentially(video_paths: List[str], embedding_model: EmbeddingModel, 
                                num_frames: int = 16) -> Dict[str, np.ndarray]:
    """Process videos sequentially (one at a time) and extract embeddings."""
    embeddings_dict = {}
    
    # Process videos one by one
    for path in tqdm(video_paths, desc="Processing videos sequentially"):
        # Check if video file exists
        if not os.path.exists(path):
            print(f"Warning: Video file not found: {path}")
            continue
        
        try:
            # Extract embedding for single video
            embedding = embedding_model.encode_video(
                [path], 
                num_frames=num_frames,
                batch_size=1
            )
            
            # Store embedding in dictionary (keep as numpy array for pickle)
            embeddings_dict[path] = embedding[0]
            
        except Exception as e:
            print(f"Error processing video {path}: {str(e)}")
            continue
    
    print(f"\nSuccessfully processed {len(embeddings_dict)} out of {len(video_paths)} videos")
    return embeddings_dict


def save_embeddings(embeddings_dict: Dict[str, np.ndarray], output_path: str):
    """Save embeddings dictionary to pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    
    print(f"Saved {len(embeddings_dict)} embeddings to {output_path}")


def merge_split_embeddings(person: str, num_splits: int):
    """Merge split embedding files into a single file."""
    print(f"\n=== Merging {num_splits} split files for {person} ===")
    
    merged_embeddings = {}
    
    for split_id in range(num_splits):
        split_file = f"output/metadata/visual_memory/{person}/visual_embeddings_split_{split_id}.pkl"
        
        if not os.path.exists(split_file):
            print(f"Warning: Split file not found: {split_file}")
            continue
        
        try:
            with open(split_file, 'rb') as f:
                split_embeddings = pickle.load(f)
            
            print(f"Loaded {len(split_embeddings)} embeddings from split {split_id}")
            merged_embeddings.update(split_embeddings)
            
        except Exception as e:
            print(f"Error loading split {split_id}: {str(e)}")
            continue
    
    if not merged_embeddings:
        print("Error: No embeddings to merge")
        return False
    
    # Save merged file
    output_file = f"output/metadata/visual_memory/{person}/visual_embeddings.pkl"
    try:
        save_embeddings(merged_embeddings, output_file)
        print(f"Successfully merged {len(merged_embeddings)} total embeddings")
        
        # Optionally remove split files after successful merge
        print("\nRemoving split files...")
        for split_id in range(num_splits):
            split_file = f"output/metadata/visual_memory/{person}/visual_embeddings_split_{split_id}.pkl"
            if os.path.exists(split_file):
                os.remove(split_file)
                print(f"Removed {split_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving merged embeddings: {str(e)}")
        return False


def process_caption_dir(
    caption_dir: str,
    output_dir: str,
    embedding_model: EmbeddingModel,
    num_frames: int = 16,
    video_ids: Optional[List[str]] = None,
):
    """Process caption subdirectories, producing per-video embeddings keyed by time."""
    if video_ids is None:
        video_dirs = sorted(
            d for d in os.listdir(caption_dir)
            if os.path.isdir(os.path.join(caption_dir, d))
        )
    else:
        video_dirs = list(video_ids)
    if not video_dirs:
        print(f"No video subdirectories found in {caption_dir}")
        return

    print(f"Processing visual embeddings for {len(video_dirs)} videos...")
    for video_id in tqdm(video_dirs, desc="Visual embeddings"):
        base_json = os.path.join(caption_dir, video_id, "10sec.json")
        if not os.path.exists(base_json):
            print(f"  Skipping {video_id}: no 10sec.json found")
            continue

        out_pkl = os.path.join(output_dir, video_id, "visual_embeddings.pkl")
        if os.path.exists(out_pkl):
            print(f"  Skipping {video_id}: visual_embeddings.pkl already exists")
            continue

        entries = load_video_entries(base_json)
        if not any('video_path' in entry for entry in entries):
            continue

        embeddings_dict: Dict[str, np.ndarray] = {}

        for entry in entries:
            vp = entry.get('video_path', '')
            start_time = str(entry.get('start_time', ''))
            end_time = str(entry.get('end_time', ''))
            if not vp or not os.path.exists(vp):
                continue

            start_sec = _time_str_to_seconds(start_time)
            end_sec = _time_str_to_seconds(end_time)
            key = start_time

            video_spec = {"video": vp, "video_start": start_sec, "video_end": end_sec}
            try:
                embedding = embedding_model.encode_video(
                    [video_spec], nframes=num_frames, batch_size=1,
                )
                embeddings_dict[key] = embedding[0]
            except Exception as e:
                print(f"  Error encoding {video_id} segment {start_time}-{end_time}: {e}")

        if embeddings_dict:
            os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
            save_embeddings(embeddings_dict, out_pkl)

    print("Visual embedding extraction complete.")


def _time_str_to_seconds(time_str: str) -> float:
    """Convert HHMMSSFF time string to seconds."""
    time_str = time_str.zfill(8)
    hours = int(time_str[0:2])
    minutes = int(time_str[2:4])
    seconds = int(time_str[4:6])
    return float(hours * 3600 + minutes * 60 + seconds)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Extract visual embeddings from videos')
    parser.add_argument('--split_id', type=int, default=None, help='Split ID for parallel processing (0-indexed).')
    parser.add_argument('--num_splits', type=int, default=1, help='Total number of splits for parallel processing')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to extract from each video')
    parser.add_argument('--person', type=str, default='A1_JAKE', help='Person to process (EgoLife mode)')
    parser.add_argument('--merge', action='store_true', help='Merge split files instead of processing videos')
    parser.add_argument('--input-json', type=str, default=None, help='Path to input caption JSON (overrides --person path)')
    parser.add_argument('--output-path', type=str, default=None, help='Path to output pickle file (overrides --person path)')
    parser.add_argument('--caption-dir', type=str, default=None, help='Caption root dir with {videoID}/ subdirs (per-video mode)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output root dir for per-video embeddings')

    args = parser.parse_args()

    # Per-video mode: process all video subdirectories
    if args.caption_dir:
        out_dir = args.output_dir or "output/metadata/visual_memory"
        embedding_model = EmbeddingModel(vis_model_name="VLM2Vec/VLM2Vec-V2.0")
        embedding_model.load_model(model_type="vision")
        process_caption_dir(args.caption_dir, out_dir, embedding_model, num_frames=args.num_frames)
        return

    name = args.person

    if args.merge:
        if args.num_splits <= 1:
            print("Error: --num_splits must be > 1 for merging")
            return
        merge_split_embeddings(name, args.num_splits)
        return

    input_json = args.input_json or f"data/EgoLife/EgoLifeCap/{name}/{name}_30sec.json"

    if args.split_id is not None:
        output_pickle = args.output_path or f"output/metadata/visual_memory/{name}/visual_embeddings_split_{args.split_id}.pkl"
        print(f"\n=== Processing {name} (split {args.split_id + 1}/{args.num_splits}) ===")
    else:
        output_pickle = args.output_path or f"output/metadata/visual_memory/{name}/visual_embeddings.pkl"
        print(f"\n=== Processing {name} (all videos) ===")

    if not os.path.exists(input_json):
        print(f"Error: Input file not found: {input_json}")
        return

    output_dir = os.path.dirname(output_pickle)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    embedding_model = EmbeddingModel(vis_model_name="VLM2Vec/VLM2Vec-V2.0")
    embedding_model.load_model(model_type="vision")

    try:
        video_paths = load_video_paths(input_json)
    except Exception as e:
        print(f"Error loading video paths: {str(e)}")
        return

    if not video_paths:
        print("No video paths found in input file")
        return

    if args.split_id is not None:
        total_videos = len(video_paths)
        videos_per_split = (total_videos + args.num_splits - 1) // args.num_splits
        start_idx = args.split_id * videos_per_split
        end_idx = min(start_idx + videos_per_split, total_videos)
        video_paths = video_paths[start_idx:end_idx]
        print(f"Split {args.split_id}: Processing videos {start_idx} to {end_idx-1} ({len(video_paths)} videos)")
    else:
        print(f"Processing all {len(video_paths)} videos")

    print("Processing videos sequentially (one at a time)...")
    embeddings_dict = process_videos_sequentially(
        video_paths,
        embedding_model,
        num_frames=args.num_frames,
    )

    if not embeddings_dict:
        print("No embeddings were extracted")
        return

    try:
        save_embeddings(embeddings_dict, output_pickle)
        print(f"Processing completed successfully!")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")


if __name__ == "__main__":
    main()
