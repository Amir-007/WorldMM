#!/usr/bin/env python3
"""
Extract Named Entities and Triples from caption text using OpenIE,
then reformat the results into a timestamp-based structure.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from worldmm.memory.episodic.openie import OpenIE
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import LLMModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_caption_data(json_file: str) -> List[Dict]:
    """Load caption data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_text_passages(caption_data: List[Dict]) -> List[str]:
    """Extract text passages from caption data."""
    passages = []
    for item in caption_data:
        if 'text' in item:
            passages.append(item['text'])
    return passages


def create_episodic_triples_results(caption_data: List[Dict], 
                                  openie_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the episodic triples results dictionary.
    
    Args:
        caption_data: List of caption dictionaries with 'text', 'date', 'end_time', 'video_path' fields
        openie_data: Dictionary containing 'triple_results' from OpenIE processing
    
    Returns:
        Dictionary with 'episodic_triples' and 'raw_video' keys
    """
    if 'triple_results' not in openie_data:
        raise ValueError("OpenIE data must contain 'triple_results' key")
    
    episodic_triples = {}
    raw_video = {}
    
    # Process each caption individually (one-by-one)
    for caption_item in caption_data:
        # Create timestamp key from each item: date[-1] + end_time.zfill(8)
        timestamp = caption_item['date'][-1] + caption_item['end_time'].zfill(8)
        
        # Create hash to match with OpenIE results
        text_hash = compute_mdhash_id(caption_item['text'], prefix="chunk-")
        
        # Get triples for this caption - must exist in OpenIE results
        if text_hash not in openie_data['triple_results']:
            print(f"Warning: Text hash {text_hash} not found in OpenIE results for text: {caption_item['text'][:100]}...")
            # Store empty lists for missing entries
            episodic_triples[timestamp] = []
            raw_video[timestamp] = [caption_item['video_path']]
            continue
        
        # Store results for this individual caption
        episodic_triples[timestamp] = openie_data['triple_results'][text_hash]
        raw_video[timestamp] = caption_item['video_path']
    
    return {
        'episodic_triples': episodic_triples,
        'raw_video': raw_video
    }


def run_episodic_triples(input_file: str, output_dir: str, model_name: str = "gpt-5-mini",
                         llm_model: LLMModel = None):
    """Core logic for episodic triple extraction, usable from CLI or as a library call."""
    os.makedirs(output_dir, exist_ok=True)

    caption_data = load_caption_data(input_file)
    logger.info(f"Loaded {len(caption_data)} caption entries from {input_file}")

    text_passages = extract_text_passages(caption_data)
    logger.info(f"Extracted {len(text_passages)} text passages")

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    openie_processor = OpenIE(llm_model)

    logger.info("Processing with OpenIE...")
    ner_results, triple_results = openie_processor.batch_openie(text_passages, output_dir=output_dir)

    total_entities = sum(len(result) for result in ner_results.values())
    total_triples = sum(len(result) for result in triple_results.values())
    logger.info(f"Total unique entities extracted: {total_entities}")
    logger.info(f"Total triples extracted: {total_triples}")

    openie_results_file = os.path.join(output_dir, f"openie_results_{model_name}.json")

    if not os.path.exists(openie_results_file):
        logger.error(f"OpenIE results file not found: {openie_results_file}")
        return

    with open(openie_results_file, 'r', encoding='utf-8') as f:
        openie_data = json.load(f)

    logger.info(f"Loaded OpenIE results with {len(openie_data.get('triple_results', {}))} text chunks")

    results = create_episodic_triples_results(caption_data, openie_data)

    total_triples_reformat = sum(len(triples) for triples in results['episodic_triples'].values())
    avg_triples = total_triples_reformat / len(results['episodic_triples']) if results['episodic_triples'] else 0
    logger.info(f"Total episodic triples: {total_triples_reformat}")
    logger.info(f"Average triples per timestamp: {avg_triples:.2f}")

    output_file = os.path.join(output_dir, f"episodic_triple_results_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully saved episodic triples results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract episodic triples from captions using OpenIE.")
    parser.add_argument("--caption-file", type=str, default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json", help="Path to caption JSON file.")
    parser.add_argument("--output-dir", type=str, default="output/metadata/episodic_memory/A1_JAKE", help="Output directory for results.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name.")
    args = parser.parse_args()

    run_episodic_triples(args.caption_file, args.output_dir, args.model)


if __name__ == "__main__":
    main()
