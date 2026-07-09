#!/usr/bin/env python3
"""
Extract semantic triples from episodic triples.
Groups caption data into chunks, retrieves corresponding OpenIE results,
and extracts semantic knowledge from them.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from worldmm.memory.semantic import SemanticExtraction
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.llm import LLMModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_PERIOD = 10

def load_caption_data(json_file: str) -> List[Dict]:
    """Load caption data from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_openie_results(json_file: str) -> Dict[str, Any]:
    """Load OpenIE results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def group_captions_and_get_openie_triples(caption_data: List[Dict], openie_data: Dict[str, Any], period: int = DEFAULT_PERIOD) -> Dict[str, List[List[str]]]:
    """
    Group caption data into chunks of `period` and retrieve corresponding OpenIE triples.
    
    Args:
        caption_data: List of caption dictionaries with 'text', 'date', 'end_time' fields
        openie_data: Dictionary containing 'triple_results' from OpenIE processing
        period: Number of captions per group
    
    Returns:
        Dictionary mapping group keys to lists of episodic triples
    """
    if 'triple_results' not in openie_data:
        raise ValueError("OpenIE data must contain 'triple_results' key")
    
    episodic_triples_batch = {}
    
    for i in range(0, len(caption_data), period):
        chunk = caption_data[i:i+period]
        
        # Create key from the last item in the group: date[-1] + end_time.zfill(8)
        last_item = chunk[-1]
        group_key = last_item['date'][-1] + last_item['end_time'].zfill(8)
        
        # Collect all triples for this group from OpenIE results
        group_triples = []
        
        for caption_item in chunk:
            # Create hash to match with OpenIE results
            text_hash = compute_mdhash_id(caption_item['text'], prefix="chunk-")
            
            # Get triples for this caption - must exist in OpenIE results
            if text_hash not in openie_data['triple_results']:
                raise ValueError(f"Text hash {text_hash} not found in OpenIE results for text: {caption_item['text'][:100]}...")
            group_triples.extend(openie_data['triple_results'][text_hash])
        
        episodic_triples_batch[group_key] = group_triples
    
    return episodic_triples_batch


def run_semantic_extraction(caption_file: str, openie_file: str, output_dir: str, model_name: str = "gpt-5-mini", llm_model: LLMModel = None, period: int = DEFAULT_PERIOD):
    """Core logic for semantic triple extraction, usable from CLI or as a library call."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(caption_file):
        logger.error(f"Caption file not found: {caption_file}")
        return
    caption_data = load_caption_data(caption_file)
    logger.info(f"Loaded {len(caption_data)} caption entries")

    if not os.path.exists(openie_file):
        logger.error(f"OpenIE results file not found: {openie_file}")
        return
    openie_data = load_openie_results(openie_file)
    logger.info(f"Loaded OpenIE results with {len(openie_data.get('triple_results', {}))} chunks")

    logger.info(f"Grouping captions into chunks of {period}...")
    try:
        episodic_triples_batch = group_captions_and_get_openie_triples(caption_data, openie_data, period=period)
        logger.info(f"Created {len(episodic_triples_batch)} caption groups")

        total_triples = sum(len(triples) for triples in episodic_triples_batch.values())
        avg_triples = total_triples / len(episodic_triples_batch) if episodic_triples_batch else 0
        logger.info(f"Total episodic triples: {total_triples}")
        logger.info(f"Average triples per group: {avg_triples:.2f}")
    except Exception as e:
        logger.error(f"Error grouping captions and extracting episodic triples: {e}")
        return

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    semantic_extraction = SemanticExtraction(llm_model)

    logger.info("Processing with Semantic Extraction...")
    semantic_triples_results, _ = semantic_extraction.batch_semantic_extraction(
        episodic_triples_batch,
        output_dir=output_dir,
    )

    total_semantic_triples = sum(len(result) for result in semantic_triples_results.values())
    logger.info(f"Total semantic triples extracted: {total_semantic_triples}")
    logger.info(f"Results saved to: {output_dir}/semantic_extraction_results_{model_name}.json")


def main():
    parser = argparse.ArgumentParser(description="Extract semantic triples from episodic triples.")
    parser.add_argument("--caption-file", type=str, default="data/EgoLife/EgoLifeCap/A1_JAKE/A1_JAKE_30sec.json", help="Path to caption JSON file.")
    parser.add_argument("--openie-file", type=str, default="output/metadata/episodic_memory/A1_JAKE/openie_results_gpt-5-mini.json", help="Path to OpenIE results JSON file.")
    parser.add_argument("--output-dir", type=str, default="output/metadata/semantic_memory/A1_JAKE", help="Output directory for results.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name.")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD, help="Number of captions per group.")
    args = parser.parse_args()

    run_semantic_extraction(
        args.caption_file, args.openie_file, args.output_dir,
        model_name=args.model, period=args.period,
    )


if __name__ == "__main__":
    main()
