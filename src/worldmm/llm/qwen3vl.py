"""
Qwen3VL Model Wrapper with comprehensive video and image processing capabilities.
"""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from .utils import dynamic_retry_decorator
from ..common.jsonrepair import salvage_json
from ..common.schema_coercion import coerce_to_schema

# Configure logging
logger = logging.getLogger(__name__)

# When structured output fails to parse, retrying with identical settings is
# pointless: generation is greedy by default, so it replays byte for byte.
# Each successive attempt therefore changes something. Sampling breaks the tie;
# a larger budget addresses genuine truncation.
# Measured over 14,496 calls on Eureka: tier 2 recovered 232 of 473 failures
# (49%), while a token-doubling tier 3 recovered 1 of 241 - the budget is
# already generous at 4096, so the residue is malformed output, not truncation.
# Tier 3 is therefore a hotter, more diverse sample rather than a bigger one.
STRUCTURED_RETRY_PLAN: List[Dict[str, Any]] = [
    {},                                                        # as configured
    {"do_sample": True, "temperature": 0.7, "top_p": 0.9},     # break the replay
    {"do_sample": True, "temperature": 1.0, "top_p": 0.95},    # try harder to differ
]

# Model configuration
MODEL_DICT = {
    "qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-30b": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
}


class Qwen3VLModelError(Exception):
    """Custom exception for Qwen3VL model operations."""
    pass


class Qwen3VLModel:
    """
    Qwen3VL model wrapper with video and image processing capabilities.
    """

    def __init__(
        self,
        model_name: str,
        max_retries: int = 3,
        max_size: Tuple[int, int] = (512, 512),
        max_size_video: Tuple[int, int] = (256, 256),
        quality: int = 85,
        fps: Optional[int] = None,
        nframes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Qwen3VL model wrapper.
        
        Args:
            model_name: Name of the Qwen3VL model to use
            max_retries: Maximum number of retry attempts
            max_size: Maximum size for image thumbnails
            max_size_video: Maximum size for video frames
            quality: JPEG quality for encoding (1-100)
            fps: Frames per second for video sampling
            nframes: Number of frames to sample from video
            **kwargs: Additional arguments for generation
            
        Raises:
            Qwen3VLModelError: If model initialization fails
            ValueError: If both fps and nframes are provided
        """
        # Validate parameters
        if fps is not None and nframes is not None:
            raise ValueError("Cannot provide both 'fps' and 'nframes'. Please choose one for video sampling.")
            
        if model_name not in MODEL_DICT:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_DICT.keys())}")

        # Set instance attributes
        self.model_name = MODEL_DICT[model_name]
        self.max_retries = max(1, max_retries)
        self.max_size = max_size
        self.max_size_video = max_size_video
        self.quality = max(1, min(100, quality))  # Clamp quality between 1-100
        self.fps = fps
        self.nframes = nframes
        self.kwargs = kwargs

        # Structured-output health, surfaced by the build runners. `repairs`
        # means salvage worked; `failures` means the retry plan was exhausted
        # and an empty result was returned, which is real data loss.
        self.parse_repairs = 0
        self.parse_failures = 0

        # Initialize model
        try:
            self._init_model()
            logger.info(f"Initialized Qwen3VLModel with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3VL model: {e}")
            raise Qwen3VLModelError("Failed to initialize Qwen3VL model") from e

    def _init_model(self) -> None:
        """Initialize the model and processor."""
        # Load model
        max_memory = {i: "62GiB" for i in range(torch.cuda.device_count())}
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory=max_memory,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and convert file path to Path object."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path

    def _calculate_sample_indices(self, vr: VideoReader, total_frames: int) -> List[int]:
        """Calculate which frames to sample from the video."""
        sample_indices = []
        
        if self.fps is not None:
            # Sample at specified FPS
            video_fps = vr.get_avg_fps()
            if video_fps <= 0:
                raise Qwen3VLModelError("Cannot determine video FPS")
                
            frame_interval = max(1, int(video_fps / self.fps))
            sample_indices = list(range(0, total_frames, frame_interval))
            
        elif self.nframes is not None:
            # Sample fixed number of frames
            if self.nframes <= 0:
                raise ValueError("nframes must be a positive integer")
                
            if self.nframes >= total_frames:
                sample_indices = list(range(total_frames))
            else:
                # Evenly distribute frames across video duration
                indices = [int(i * (total_frames - 1) / (self.nframes - 1)) for i in range(self.nframes)]
                sample_indices = sorted(list(set(indices)))
                
        else:
            # Default: sample at 1 FPS
            logger.warning("No fps or nframes specified, defaulting to 1 FPS sampling")
            video_fps = vr.get_avg_fps()
            frame_interval = max(1, int(video_fps)) if video_fps > 0 else 30
            sample_indices = list(range(0, total_frames, frame_interval))

        # Ensure we have at least the first and last frame
        if sample_indices and sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices and sample_indices[-1] != total_frames - 1:
            sample_indices.append(total_frames - 1)
            
        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))
        
        logger.debug(f"Sampling {len(sample_indices)} frames from {total_frames} total frames")
        return sample_indices

    def _process_image(self, image_input: Any) -> Image.Image:
        """Process a single image input with resizing."""
        # Load image
        if isinstance(image_input, (str, Path)):
            path = self._validate_file_path(image_input)
            img = Image.open(path)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        # Ensure RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize maintaining aspect ratio
        img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        
        return img

    def _process_video(self, video_path: Union[str, Path]) -> List[Image.Image]:
        """Extract video frames as PIL Image objects with resizing."""
        video_path = self._validate_file_path(video_path)
        
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                raise Qwen3VLModelError(f"Video file appears to be empty or corrupted: {video_path}")

            # Determine sampling strategy
            sample_indices = self._calculate_sample_indices(vr, total_frames)
            
            # Extract frames as PIL Images
            frames = []
            for idx in sample_indices:
                try:
                    frame = vr[idx].asnumpy()
                    img = Image.fromarray(frame)
                    
                    # Ensure RGB mode
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Resize maintaining aspect ratio
                    img.thumbnail(self.max_size_video, Image.Resampling.LANCZOS)
                    
                    frames.append(img)
                except Exception as e:
                    logger.warning(f"Failed to process frame {idx}: {e}")
                    continue

            return frames
            
        except Exception as e:
            logger.error(f"Failed to extract video frames {video_path}: {e}")
            raise Qwen3VLModelError(f"Failed to extract video frames: {e}") from e

    def _preprocess_prompt(self, prompt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess prompt to convert it to Qwen3VL message format.
        
        Args:
            prompt: Conversation prompt in our internal format
            
        Returns:
            Prompt in Qwen3VL message format
        """
        prompt_copy = copy.deepcopy(prompt)

        for item in prompt_copy:
            if isinstance(item["content"], str):
                item["content"] = [{"type": "text", "text": item["content"]}]

        # Collect items with content to process in parallel
        content_items = [(i, item) for i, item in enumerate(prompt_copy)]
            
        # Process content items in parallel using ThreadPoolExecutor
        max_workers = min(len(content_items), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_content, item["content"]): i 
                for i, item in content_items
            }
            
            # Wait for all content processing to complete
            for future in as_completed(future_to_index):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing content: {e}")
                    
        return prompt_copy

    def _process_content(self, content: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """Process content by converting image and video references."""
        if isinstance(content, str):
            return

        if isinstance(content, dict):
            content = [content]

        if not isinstance(content, list):
            raise ValueError("content must be a str, dict, or list")

        # Collect image/video tasks
        media_tasks = []
        for i, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(f"Content item must be a dict, got {type(item)}")

            t = item.get("type")
            if t == "text" and "text" in item:
                # Keep text items as-is
                continue
            elif t == "image" and "image" in item:
                media_tasks.append((i, "image", item["image"]))
            elif t == "video" and "video" in item:
                media_tasks.append((i, "video", item["video"]))
            else:
                raise ValueError(f"Unsupported media item at index {i}: {item}")

        if not media_tasks:
            return

        # Process image/video tasks in parallel
        max_workers = min(len(media_tasks), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            for task in media_tasks:
                i, media_type, file = task
                if media_type == "image":
                    fut = executor.submit(self._process_image, file)
                elif media_type == "video":
                    fut = executor.submit(self._process_video, file)
                else:
                    raise ValueError(f"Unsupported media type at index {i}: {media_type}")
                future_to_task[fut] = task

            # Collect results
            results = []
            for fut in as_completed(future_to_task):
                i, media_type, _ = future_to_task[fut]
                try:
                    if media_type == "image":
                        processed_image = fut.result()
                        results.append((i, "image", processed_image))
                    elif media_type == "video":
                        video_frames = fut.result()
                        results.append((i, "video", video_frames))
                except Exception as e:
                    logger.error(f"Failed to process {media_type} content at index {i}: {e}")
                    results.append((i, "error", None))

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])

            # Apply results to content with proper shifting
            shift = 0
            for i, media_type, result in results:
                if media_type == "error":
                    continue
                elif media_type == "image":
                    content[i + shift] = {"type": "image", "image": result}
                elif media_type == "video":
                    # For Qwen3VL, we convert video frames to image items
                    image_items = [{"type": "image", "image": frame} for frame in result]
                    content[i + shift:i + shift + 1] = image_items
                    shift += len(image_items) - 1

    def _normalize_prompt(self, prompt: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Coerce string prompts into a single user message."""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return prompt

    @dynamic_retry_decorator
    def generate(self, prompt: Union[str, List[Dict[str, Any]]], text_format: Optional[type] = None, **kwargs) -> Any:
        """
        Generate completion for a single prompt.
        
        Args:
            prompt: Conversation prompt in our internal format or raw string
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call
            
        Returns:
            Generated response string or parsed object if text_format is provided
            
        Raises:
            Qwen3VLModelError: If generation fails
        """
        prompt_copy = copy.deepcopy(self._normalize_prompt(prompt))
        messages = self._preprocess_prompt(prompt_copy)

        if text_format is None:
            return self._generate_once(messages, **kwargs)

        # Structured output: escalate through the retry plan rather than
        # accepting the first unparseable response. Previously any parse
        # failure returned a schema-valid *empty* object, which looked like
        # success to the retry decorator and silently cost the chunk all of
        # its triples - 13.2% of the inherited build set came back empty this
        # way, with a non-empty caption every time.
        last_raw = ""
        for attempt, plan_entry in enumerate(STRUCTURED_RETRY_PLAN, start=1):
            overrides = dict(plan_entry)   # never mutate the shared plan
            multiplier = overrides.pop("_max_new_tokens_multiplier", None)
            attempt_kwargs = dict(kwargs)
            attempt_kwargs.update(overrides)
            if multiplier:
                budget = attempt_kwargs.get("max_new_tokens") \
                    or self.kwargs.get("max_new_tokens") or 2048
                attempt_kwargs["max_new_tokens"] = int(budget * multiplier)

            last_raw = self._generate_once(messages, **attempt_kwargs)
            value, how = self._parse_structured_response(last_raw, text_format)

            if how != "failed":
                if how not in ("clean", "fenced"):
                    self.parse_repairs += 1
                    logger.warning(
                        "Recovered structured output by %s on attempt %d/%d",
                        how, attempt, len(STRUCTURED_RETRY_PLAN),
                    )
                return value

            logger.warning(
                "Unparseable structured output on attempt %d/%d; raw head: %r",
                attempt, len(STRUCTURED_RETRY_PLAN), last_raw[:200],
            )

        # Every attempt failed. Degrade to empty so a single pathological chunk
        # cannot abort a multi-day run, but make it loud and countable rather
        # than indistinguishable from a genuinely empty result.
        self.parse_failures += 1
        logger.error(
            "Giving up on structured output after %d attempts; returning empty. "
            "Raw head: %r", len(STRUCTURED_RETRY_PLAN), last_raw[:200],
        )
        return self._empty_default(text_format)

    def _generate_once(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Run the model once and return the decoded text."""
        try:
            # Prepare text for Qwen3VL processor
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info (images and videos)
            # process_vision_info returns (image_inputs, video_inputs, video_kwargs)
            vision_outputs = process_vision_info(messages)
            image_inputs = vision_outputs[0] if len(vision_outputs) > 0 else None
            video_inputs = vision_outputs[1] if len(vision_outputs) > 1 else None
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # Merge kwargs with instance kwargs (call-specific kwargs take precedence)
            gen_kwargs = {**self.kwargs, **kwargs}
            
            # Set default generation parameters if not specified
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 2048
            if "repetition_penalty" not in gen_kwargs:
                gen_kwargs["repetition_penalty"] = 1.3
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            
            # Generate
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip() if output_text else ""

        except Exception as e:
            logger.error(f"Qwen3VL generation error: {e}")
            raise Qwen3VLModelError(f"Failed to generate completion: {e}") from e

    def generate_batch(self, batch_prompts: List[Union[str, List[Dict[str, Any]]]], text_format: Optional[type] = None, **kwargs) -> List[Any]:
        """
        Process multiple prompts in batch.
        
        Args:
            batch_prompts: List of conversation prompts
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call
            
        Returns:
            List of response strings or parsed objects if text_format is provided
            
        Raises:
            Qwen3VLModelError: If batch generation fails
        """
        if not batch_prompts:
            return []
        
        # Process prompts in parallel using ThreadPoolExecutor
        max_workers = min(len(batch_prompts), (os.cpu_count() or 1) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda p: self.generate(self._normalize_prompt(p), text_format=text_format, **kwargs),
                batch_prompts
            ))
        
        return results

    def generate_batch_parallel(
        self,
        batch_prompts: List[Union[str, List[Dict[str, Any]]]],
        max_workers: Optional[int] = None,
        text_format: Optional[type] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process multiple prompts in parallel using ThreadPoolExecutor.
        
        This is an alias for generate_batch with explicit max_workers parameter.
        
        Args:
            batch_prompts: List of conversation prompts
            max_workers: Maximum number of parallel workers (defaults to min(32, len(batch_prompts)))
            text_format (optional): Pydantic model or other structure for parsing the response
            **kwargs: Additional arguments for the generation call
            
        Returns:
            List of response strings or parsed objects if text_format is provided, in the same order as input
            
        Raises:
            Qwen3VLModelError: If batch generation fails
        """
        if not batch_prompts:
            return []
        
        if max_workers is None:
            max_workers = min(32, len(batch_prompts))
        
        results: List[Any] = [""] * len(batch_prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.generate, prompt, text_format, **kwargs): i
                for i, prompt in enumerate(batch_prompts)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process prompt at index {index}: {e}")
                    raise Qwen3VLModelError(f"Batch generation failed at index {index}") from e
        
        return results

    @staticmethod
    def _empty_default(fmt):
        """A schema-valid instance with every field empty."""
        try:
            fields = getattr(fmt, 'model_fields', {})
            return fmt.model_validate({fname: [] for fname in fields})
        except Exception:
            return None

    def _parse_structured_response(self, response: str, text_format: type) -> Tuple[Any, str]:
        """
        Parse a response into the given Pydantic model.

        Returns:
            (value, how) where `how` is "clean"/"fenced" for output that parsed
            as generated, "sliced"/"repaired" when salvage was needed, and
            "failed" when nothing usable could be recovered.

        Crucially this reports failure instead of substituting an empty result:
        the caller escalates through the retry plan, which is what a degraded
        generation actually needs.
        """
        try:
            json_data, how = salvage_json(response)
            if how == "failed" or json_data is None:
                return None, "failed"

            json_data = coerce_to_schema(
                json_data, list(getattr(text_format, 'model_fields', {}).keys())
            )

            if hasattr(text_format, 'model_validate'):
                return text_format.model_validate(json_data), how
            # Fallback for older Pydantic versions
            elif hasattr(text_format, 'parse_obj'):
                return text_format.parse_obj(json_data), how
            else:
                logger.warning(f"Unsupported text_format type: {text_format}")
                raise ValueError("text_format must be a Pydantic model class")

        except Exception as e:
            # An unforeseen shape is a failed parse, not an empty answer. The
            # caller retries; only after the plan is exhausted does it degrade.
            logger.warning(
                "Structured response did not match the schema (%s). Raw head: %r",
                e, response[:200],
            )
            return None, "failed"


# Convenience function
def create_qwen3vl_model(
    model_name: str,
    **kwargs
) -> Qwen3VLModel:
    """Create a Qwen3VL model instance with convenient defaults."""
    return Qwen3VLModel(model_name=model_name, **kwargs)
