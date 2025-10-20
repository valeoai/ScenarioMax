"""
Stage 3: Format - Convert unified scenarios to target formats.

This module provides format converters for transforming unified scenarios
into various output formats (TFExample, JSON, Puffer, etc.).

Submodules:
- tfexample: TensorFlow Example format converter
- json: JSON format converter (GPUDrive)
- puffer: Puffer simulator format converter
"""

__all__ = ["tfexample", "json", "puffer"]
