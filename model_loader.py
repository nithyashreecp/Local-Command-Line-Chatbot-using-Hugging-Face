# model_loader.py
"""
Load a Hugging Face text-generation pipeline.
"""

from transformers import pipeline
import torch

def detect_device(prefer_gpu: bool = False):
    """
    Return device id for transformers.pipeline:
      -1 means CPU
      >=0 means GPU index (e.g., 0)
    prefer_gpu: if True and CUDA is available returns 0, else -1
    """
    if prefer_gpu and torch.cuda.is_available():
        return 0
    return -1

def load_text_generation_pipeline(model_name: str = "distilgpt2", device_id: int = None):
    """
    Loads and returns a Hugging Face text-generation pipeline.

    Args:
      model_name: model checkpoint (default "distilgpt2")
      device_id: -1 for CPU, >=0 for GPU index. If None, auto-detect CPU/GPU.
    Returns:
      generator: pipeline object (callable)
    """
    if device_id is None:
        device_id = detect_device(prefer_gpu=False)  # default: CPU

    # Create pipeline. This will download model files the first time you run it.
    generator = pipeline("text-generation", model=model_name, tokenizer=model_name, device=device_id)
    return generator
