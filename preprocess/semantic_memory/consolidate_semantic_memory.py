#!/usr/bin/env python3
"""
Consolidate semantic triples across timestamps.
Loads semantic extraction results and applies semantic consolidation.
"""

import argparse
import json
import os
from typing import Dict, Any
from tqdm import tqdm

from worldmm.memory.semantic import SemanticConsolidation
from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_semantic_extraction_results(json_file: str) -> Dict[str, Any]:
    """Load semantic extraction results from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_semantic_consolidation(semantic_file: str, output_dir: str, model_name: str = "gpt-5-mini", llm_model: LLMModel = None, embedding_model: EmbeddingModel = None):
    """Core logic for semantic consolidation, usable from CLI or as a library call."""
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(semantic_file):
        logger.error(f"Semantic extraction results file not found: {semantic_file}")
        return

    semantic_results = load_semantic_extraction_results(semantic_file)
    logger.info(f"Loaded semantic extraction results with {len(semantic_results.get('semantic_triples', {}))} timestamps")

    total_triples_before = sum(
        len(triples) for triples in semantic_results.get('semantic_triples', {}).values()
    )
    logger.info(f"Total semantic triples before consolidation: {total_triples_before}")

    if embedding_model is None:
        embedding_model = EmbeddingModel(text_model_name="Qwen/Qwen3-Embedding-4B")
        embedding_model.load_model(model_type="text")

    if llm_model is None:
        llm_model = LLMModel(model_name=model_name)

    semantic_consolidation = SemanticConsolidation(llm_model, embedding_model)

    logger.info("Processing with Semantic Consolidation...")

    timestamps = sorted(semantic_results.get('semantic_triples', {}).keys())

    accumulated_semantic_triples = []
    accumulated_episodic_evidence = []
    timestamped_results = {}

    for i, timestamp in tqdm(enumerate(timestamps), total=len(timestamps)):
        logger.info(f"Processing timestamp {timestamp} ({i+1}/{len(timestamps)})")

        current_triples = semantic_results['semantic_triples'].get(timestamp, [])
        # current_evidence = semantic_results['episodic_evidence'].get(timestamp, [])
        #
        # transformed_current_evidence = []
        # for evidence_list in current_evidence:
        #     transformed_list = [f"{timestamp}_{idx}" for idx in evidence_list]
        #     transformed_current_evidence.append(transformed_list)
        
        transformed_current_evidence = [[] for _ in current_triples]

        existing_results = (accumulated_semantic_triples.copy(), accumulated_episodic_evidence.copy())
        new_results = (current_triples, transformed_current_evidence)

        consolidated_triples, consolidated_evidence, triples_to_remove = semantic_consolidation.batch_semantic_consolidation(
            existing_results, new_results
        )

        logger.debug(f"  Removing {len(triples_to_remove)} existing triples")

        triples_to_remove_set = set()
        for triple, evidence in triples_to_remove:
            triples_to_remove_set.add((tuple(triple), tuple(evidence)))

        new_accumulated_triples = []
        new_accumulated_evidence = []
        for acc_triple, acc_evidence in zip(accumulated_semantic_triples, accumulated_episodic_evidence):
            triple_key = (tuple(acc_triple), tuple(acc_evidence))
            if triple_key not in triples_to_remove_set:
                new_accumulated_triples.append(acc_triple)
                new_accumulated_evidence.append(acc_evidence)

        accumulated_semantic_triples = new_accumulated_triples
        accumulated_episodic_evidence = new_accumulated_evidence

        accumulated_semantic_triples.extend(consolidated_triples)
        accumulated_episodic_evidence.extend(consolidated_evidence)

        timestamped_results[timestamp] = {
            "consolidated_semantic_triples": accumulated_semantic_triples,
        }

    total_triples_after = len(accumulated_semantic_triples)
    logger.info(f"Total semantic triples after consolidation: {total_triples_after}")
    if total_triples_before > 0:
        logger.info(f"Reduction in triples: {total_triples_before - total_triples_after} "
                     f"({((total_triples_before - total_triples_after) / total_triples_before * 100):.1f}%)")

    output_file = os.path.join(output_dir, f"semantic_consolidation_results_{model_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timestamped_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results have been saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate semantic triples across timestamps.")
    parser.add_argument("--semantic-file", type=str, default="output/metadata/semantic_memory/A1_JAKE/semantic_extraction_results_gpt-5-mini.json", help="Path to semantic extraction results JSON file.")
    parser.add_argument("--output-dir", type=str, default="output/metadata/semantic_memory/A1_JAKE", help="Output directory for results.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="LLM model name.")
    args = parser.parse_args()

    run_semantic_consolidation(args.semantic_file, args.output_dir, model_name=args.model)


if __name__ == "__main__":
    main()
