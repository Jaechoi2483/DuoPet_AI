"""
Model Adapters for DuoPet AI Health Diagnosis System
"""

from .base_adapter import ModelAdapter
from .eye_disease_adapter import EyeDiseaseAdapter
from .bcs_adapter import BCSAdapter
from .skin_disease_adapter import SkinDiseaseAdapter

__all__ = ['ModelAdapter', 'EyeDiseaseAdapter', 'BCSAdapter', 'SkinDiseaseAdapter']