import pytest
import torch
from asr_service.services.model_loader import ASREngine


def test_model_loads_successfully():
    """Test that the ASR model loads without errors."""
    engine = ASREngine()

    engine.load_model()

    assert engine.is_loaded is True
    assert engine.final_pipeline is not None
    assert engine.live_pipeline is not None


def test_singleton_pattern_same_instance():
    """Test that ASREngine follows singleton pattern."""
    engine1 = ASREngine()
    engine2 = ASREngine()

    assert engine1 is engine2


def test_model_device_selection():
    """Test that model correctly selects device (MPS/CUDA/CPU)."""
    engine = ASREngine()
    engine.load_model()

    expected_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    assert engine.device == expected_device
