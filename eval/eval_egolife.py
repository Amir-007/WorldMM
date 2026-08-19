#!/usr/bin/env python3
"""
EgoLifeQA evaluation script using WorldMM unified memory system.
"""

import os
import json
import re
import argparse
from typing import Iterable, Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult, transform_timestamp


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str, valid_letters: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Pull the chosen option letter out of a free-form model response.

    Only letters in valid_letters are accepted. Without that filter, a pattern
    keyed on the word "answer" happily returns the first letter of whatever word
    follows it, so "I cannot answer this question" yields "T". Filtering also
    lets a pattern that matched junk fall through to a later pattern instead of
    returning early with a wrong letter.
    """
    t = (text or "").strip()
    if not t:
        return None
    valid = {l.upper() for l in valid_letters} if valid_letters else set("ABCD")
    patterns = [
        r"\\boxed\{\(?([A-Za-z])\)?\}",                       # $\boxed{A}$
        r"\*\*\(?([A-Za-z])\)?\*\*",                          # **(A)** or **A**
        # An explicit separator is required, so prose like "to answer this" does
        # not capture the next word's first letter.
        r"(?:final answer|correct answer|answer|correct response|option)"
        r"\s*(?:is|:|=)\s*\**\(?([A-Za-z])\)?\**",
        r"^\(?([A-Za-z])\)?[\.\):]?\s*$",                     # whole response is "A", "(A)", "A."
        r"^\(?([A-Za-z])[\.\)]",                              # "A." or "(A)" leading longer text
        r"\b([A-Za-z])[\.\)]?\s*$",                           # trailing letter, "... so B."
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, t, re.IGNORECASE | re.MULTILINE):
            letter = match.group(1).upper()
            if letter in valid:
                return letter
    return None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    """
    Evaluate if prediction matches the gold answer.
    
    Args:
        prediction: Model's prediction
        gold_letter: Correct answer letter (e.g., 'A', 'B', 'C', 'D')
        choices: Dict of answer choices
        
    Returns:
        True if prediction is correct
    """
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction, valid_letters=choices.keys())
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}")
    ]
    if pred_norm in full_patterns:
        return True

    return False


def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Find the 30s segment that contains the target timestamp.
    
    Args:
        target_timestamp: Target timestamp as integer (format: day + time.zfill(8))
        segments_30s: List of 30s segments
    
    Returns:
        Tuple of (start_time, end_time) for the matching segment, or (0, 0) if not found
    """
    for segment in segments_30s:
        date = segment.get('date', '')
        start_time_raw = segment.get('start_time', 0)
        end_time_raw = segment.get('end_time', 0)
        
        day = date.replace('DAY', '').replace('Day', '') if isinstance(date, str) else str(date)
        
        # Format times
        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue
        
        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue
        
        # Check if target timestamp falls within this segment
        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)
    
    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Parse target time from row data.
    
    Args:
        row: QA row data
        segments_30s: List of 30s segments for finding time ranges
        
    Returns:
        List of (start_time, end_time) tuples
    """
    target_time_list = []
    
    if "time" in row['target_time'] and row['target_time']["time"]:
        time_str = row['target_time']["time"]
        time_str_upper = time_str.upper()
        
        if "DAY" in time_str_upper:
            # Parse range format: "11153417DAY1_11181201"
            parts = re.split(r'DAY|Day', time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
                    
                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            # Single timestamp - find its 30s segment
            day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    elif "time_list" in row['target_time'] and row['target_time']["time_list"]:
        # Multiple timestamps
        day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
        for time_str in row['target_time']["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    return target_time_list


# Responses that mean the pipeline failed rather than the model answering badly.
# memory.py returns the second when generation raises; the eval loop returns the
# first when answer() raises outright.
FAILURE_RESPONSES = ("Error", "Unable to generate answer")


def is_failed_entry(entry: Dict[str, Any]) -> bool:
    """True when the entry records an infrastructure failure, not a model answer."""
    if entry.get("failed"):
        return True
    return str(entry.get("response", "")).strip() in FAILURE_RESPONSES


def load_checkpoint(path: str) -> List[Dict[str, Any]]:
    """
    Read completed results from a JSONL checkpoint.

    A run killed mid-write can leave a truncated final line, so unparseable
    lines are dropped rather than aborting the resume. Later entries for the
    same ID win, which matters if a question was retried.
    """
    if not os.path.exists(path):
        return []
    by_id: Dict[Any, Dict[str, Any]] = {}
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(entry, dict) and "ID" in entry:
                by_id[entry["ID"]] = entry
            else:
                skipped += 1
    if skipped:
        logger.warning("Skipped %d unparseable line(s) in %s", skipped, path)
    return list(by_id.values())


def append_checkpoint(path: str, entry: Dict[str, Any]) -> None:
    """Append one result and flush, so a crash loses at most the question in flight."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recompute every metric from the full result set.

    Derived from the results themselves rather than from running counters, so
    the numbers are identical whether the run completed in one pass or was
    resumed a dozen times.
    """
    total = len(results)
    failed = sum(1 for r in results if is_failed_entry(r))
    abstained = sum(1 for r in results if r.get("abstained") and not is_failed_entry(r))
    answered = total - abstained - failed
    correct = sum(1 for r in results if r.get("evaluate") is True)
    wrong = answered - correct
    return {
        "total": total,
        "answered": answered,
        "abstained": abstained,
        "failed": failed,
        "correct": correct,
        "wrong": wrong,
        "accuracy_answered": correct / answered if answered else 0.0,
        "accuracy_overall": correct / total if total else 0.0,
        "abstention_rate": abstained / total if total else 0.0,
        "hallucination_rate": wrong / total if total else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--retriever-model", type=str, default="qwen3vl-30b", help="LLM model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="gpt-5", help="LLM model for iterative reasoning and generating answers")
    parser.add_argument("--memory-model", type=str, default="qwen3vl-30b", help="Model used to build memory banks.")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    parser.add_argument("--metadata-dir", type=str, default="output/metadata", help="Root metadata directory containing the built memory banks")
    parser.add_argument("--enable-spatial", action="store_true", help="Load the Entity ID bank so retrieval can distinguish same-named items by location.")
    parser.add_argument("--enable-abstention", action="store_true", help="Halt and ask a clarifying question when a query is ambiguous, instead of guessing. Implies --enable-spatial.")
    parser.add_argument("--confidence-threshold", type=float, default=0.75, help="Abstain below this confidence.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip questions already present in the results file (default).")
    resume_group.add_argument("--fresh", dest="resume", action="store_false", help="Ignore any existing results and start over.")
    args = parser.parse_args()

    # Resolve every input path up front, then verify all of them exist BEFORE loading
    # any model. Model loading takes ~50 minutes, so a missing file must fail in
    # seconds rather than after the load.
    subject = args.subject
    data_dir = args.data_dir

    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    episodic_caption_dir = os.path.join(data_dir, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }
    semantic_path = os.path.join(
        args.metadata_dir, "semantic_memory", subject,
        f"semantic_consolidation_results_{args.memory_model}.json")
    visual_path = os.path.join(
        args.metadata_dir, "visual_memory", subject, "visual_embeddings.pkl")
    spatial_path = os.path.join(
        args.metadata_dir, "spatial_memory", subject, "entity_ids.json")

    # --enable-abstention is meaningless without the bank it reasons over.
    use_spatial = args.enable_spatial or args.enable_abstention

    # The run tag keeps ablation configurations in separate files. Without it a
    # later arm silently overwrites an earlier one, and the two become
    # impossible to tell apart afterwards.
    if args.enable_abstention:
        run_tag = f"abstain{args.confidence_threshold:g}"
    elif args.enable_spatial:
        run_tag = "spatial"
    else:
        run_tag = "baseline"

    run_dir = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}")
    output_stem = f"egolife_eval_{subject}_{run_tag}"
    checkpoint_path = os.path.join(run_dir, f"{output_stem}.jsonl")
    output_path = os.path.join(run_dir, f"{output_stem}.json")

    required = {
        "eval data": eval_data_path,
        "semantic memory": semantic_path,
        "visual embeddings": visual_path,
        **{f"captions {g}": f for g, f in episodic_caption_files.items()},
    }
    if use_spatial:
        required["spatial entity bank"] = spatial_path
    missing = {k: v for k, v in required.items() if not os.path.exists(v)}
    if missing:
        for k, v in missing.items():
            logger.error(f"Missing required input ({k}): {v}")
        raise SystemExit(1)
    logger.info(f"All {len(required)} required input files found")

    # Initialize models
    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    
    # Local models consume VRAM per instance, unlike API models. When retriever and
    # responder are the same local model, load once and share the instance.
    if args.retriever_model == args.respond_model:
        logger.info(f"Sharing one instance for retriever and responder ({args.retriever_model})")
        retriever_llm_model = LLMModel(
            model_name=args.retriever_model,
            fps=1,
        )
        respond_llm_model = retriever_llm_model
    else:
        retriever_llm_model = LLMModel(
            model_name=args.retriever_model,
        )
        respond_llm_model = LLMModel(
            model_name=args.respond_model,
            fps=1,
        )
    prompt_template_manager = PromptTemplateManager()

    # Initialize WorldMemory
    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
        enable_abstention=args.enable_abstention,
        confidence_threshold=args.confidence_threshold,
    )
    
    # Set retrieval top-k
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # Load data (paths were resolved and verified before model loading)
    logger.info("Loading data...")
    eval_data = load_json(eval_data_path)

    # Load 30sec captions separately for target time parsing
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])

    # Load semantic results
    semantic_results = load_json(semantic_path)
    
    # Load data into WorldMemory
    logger.info("Loading data into WorldMemory...")
    
    # Load episodic captions for all granularities
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    
    # Load semantic triples
    world_memory.load_semantic_triples(data=semantic_results)
    
    # Load visual embeddings
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    # Load the Entity ID bank
    if use_spatial:
        world_memory.load_spatial_entities(file_path=spatial_path)
        logger.info("Spatial memory enabled (abstention=%s, threshold=%.2f)", args.enable_abstention, args.confidence_threshold)

    # Evaluation loop. Results are appended to a JSONL checkpoint as each
    # question finishes, so an interrupted run keeps everything it completed.
    os.makedirs(run_dir, exist_ok=True)

    if args.resume:
        results = load_checkpoint(checkpoint_path)
        if results:
            logger.info("Resuming from %s with %d question(s) already done",
                        checkpoint_path, len(results))
    else:
        results = []
        if os.path.exists(checkpoint_path):
            backup = checkpoint_path + ".bak"
            os.replace(checkpoint_path, backup)
            logger.info("Starting fresh; previous checkpoint moved to %s", backup)

    # Failed entries are kept for diagnostics but are not treated as complete, so
    # a transient OOM is retried on the next run rather than frozen as a wrong answer.
    completed_ids = {r["ID"] for r in results if not is_failed_entry(r)}
    retryable = sum(1 for r in results if is_failed_entry(r))
    if retryable:
        logger.info("%d previously failed question(s) will be retried", retryable)
    pending = [row for row in eval_data if row["ID"] not in completed_ids]
    logger.info("Evaluating %d of %d question(s); %d already complete",
                len(pending), len(eval_data), len(completed_ids))

    for row in tqdm(pending):
        ID = row['ID']
        query_type = row['type']
        question = row['question']
        answer = row['answer']

        # Parse choices
        choices = {}
        for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
            if key in row and row[key]:
                choices[label] = row[key]

        # Parse query time
        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))
        
        # Parse target time (use 30sec captions for segment lookup)
        target_time_list = parse_target_time(row, episodic_captions_30sec)

        logger.info(f"Processing ID {ID}: {question[:50]}...")

        qa_result: Optional[QAResult] = None
        try:            
            # Answer the question
            qa_result = world_memory.answer(
                query=question,
                choices=choices,
                until_time=query_time,
            )
            
            response = qa_result.answer
            
        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        # Evaluate. An abstention is neither correct nor a hallucination: the
        # agent declined to guess. Scoring it as wrong would penalise exactly the
        # behaviour we are trying to produce, so it is excluded from accuracy
        # and counted separately.
        #
        # A generation failure (OOM, and so on) is neither. Scoring it wrong
        # would silently deflate accuracy with an infrastructure problem, so it
        # is marked failed, excluded from the metrics, and retried on resume.
        abstained = bool(qa_result.abstained) if qa_result else False
        failed = response.strip() in FAILURE_RESPONSES
        evaluate = None if (abstained or failed) else evaluate_prediction(response, answer, choices)

        # Build result entry
        result_entry = {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "evaluate": evaluate,
            "abstained": abstained,
            "failed": failed,
            "confidence": qa_result.confidence if qa_result else None,
            "ambiguity": qa_result.ambiguity if qa_result else None,
            "query_time": query_time,
            # "query_time_str": transform_timestamp(str(query_time)),
            "target_time": target_time_list,
            # "target_time_str": [
            #     (transform_timestamp(str(start)), transform_timestamp(str(end))) 
            #     for start, end in target_time_list
            # ],
        }
        results.append(result_entry)
        append_checkpoint(checkpoint_path, result_entry)

        # Running figures come from the full result set, resumed entries included.
        running = summarise(results)
        verdict = "ABSTAIN" if abstained else f"Correct: {evaluate}"
        logger.info(
            f"ID {ID} Answer: {response}, Gold: {answer}, {verdict} "
            f"// Accuracy: {running['correct']}/{running['answered']} "
            f"= {running['accuracy_answered']:.4f} "
            f"// Abstained: {running['abstained']} "
            f"// Done: {running['total']}/{len(eval_data)}"
        )

    # Save results. The JSONL checkpoint is the source of truth; this is the
    # aggregated view, rebuilt from it so both agree even after a resume.
    results.sort(key=lambda r: str(r.get("ID")))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Accuracy, abstention rate and hallucination rate are reported separately:
    # an agent that asks instead of guessing wrong has avoided a hallucination,
    # so collapsing them into one number hides the effect being measured.
    stats = summarise(results)
    summary_path = os.path.join(run_dir, f"{output_stem}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "subject": subject,
            "run_tag": run_tag,
            "retriever_model": args.retriever_model,
            "respond_model": args.respond_model,
            "memory_model": args.memory_model,
            "enable_spatial": use_spatial,
            "enable_abstention": args.enable_abstention,
            "confidence_threshold": args.confidence_threshold,
            "expected_questions": len(eval_data),
            **stats,
        }, f, indent=4)

    logger.info(f"\n{'='*50}")
    logger.info("Evaluation Complete")
    logger.info(f"Subject              : {subject}")
    logger.info(f"Run                  : {run_tag}")
    logger.info(f"Total questions      : {stats['total']} of {len(eval_data)}")
    logger.info(f"Answered             : {stats['answered']}")
    logger.info(f"Abstained            : {stats['abstained']}")
    logger.info(f"Failed (infra)       : {stats['failed']}")
    logger.info(f"Correct              : {stats['correct']}")
    logger.info(f"Wrong                : {stats['wrong']}")
    logger.info(f"Accuracy (answered)  : {stats['accuracy_answered']:.4f}")
    logger.info(f"Accuracy (overall)   : {stats['accuracy_overall']:.4f}")
    logger.info(f"Abstention rate      : {stats['abstention_rate']:.4f}")
    logger.info(f"Hallucination rate   : {stats['hallucination_rate']:.4f}")
    if stats['total'] < len(eval_data):
        logger.warning("Incomplete: %d question(s) remain. Rerun the same command to resume.",
                       len(eval_data) - stats['total'])
    logger.info(f"Results  : {output_path}")
    logger.info(f"Summary  : {summary_path}")
    logger.info(f"{'='*50}")

    # Cleanup
    world_memory.cleanup()


if __name__ == "__main__":
    main()
