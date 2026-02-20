from asr_service.services.model_loader import ASREngine
from asr_service.services.audio_processor import AudioProcessor


def get_asr_engine() -> ASREngine:
    """
    Dependency injection for ASR engine.

    Returns:
        Singleton ASREngine instance
    """
    return ASREngine()


def get_audio_processor() -> AudioProcessor:
    """
    Dependency injection for audio processor.

    Returns:
        AudioProcessor instance
    """
    return AudioProcessor()
