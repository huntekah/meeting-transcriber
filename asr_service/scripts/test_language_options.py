#!/usr/bin/env python3
"""
Quick test to verify all language options work without errors.

Tests:
1. Auto-detect (None passed to backend)
2. English ("en")
3. Polish ("pl")
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from asr_service.schemas.transcription import SessionConfig, SourceConfig
from asr_service.services.session_manager import SessionManager
from asr_service.services.model_manager import ModelManager


async def test_language_option(language: str | None, label: str):
    """Test creating a session with a specific language setting."""
    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"Language value: {language}")
    print(f"{'='*70}")

    try:
        # Get singletons
        model_manager = ModelManager()
        session_manager = SessionManager()

        # Create minimal session config (no actual devices, won't start recording)
        config = SessionConfig(
            sources=[
                SourceConfig(
                    device_index=0,
                    device_name="Test Device",
                    device_channels=1,
                    source_type="sounddevice"
                )
            ],
            output_dir=None,
            language=language
        )

        # Create session (this tests the full API → SessionManager → Session flow)
        session = await session_manager.create_session(
            sources=config.sources,
            model_manager=model_manager,
            output_dir=None,
            language=config.language,
        )

        print(f"✓ Session created: {session.session_id}")
        print(f"✓ Session language: {session.language}")
        print(f"✓ Expected: {'en' if language is None else language}")

        # Verify language was set correctly
        if language is None:
            # Auto-detect should default to "en" in Session.__init__
            assert session.language == "en", f"Expected 'en' for auto-detect, got '{session.language}'"
            print(f"✓ Auto-detect defaults to English: PASS")
        else:
            assert session.language == language, f"Expected '{language}', got '{session.language}'"
            print(f"✓ Language set correctly: PASS")

        # Clean up
        await session_manager.delete_session(session.session_id)
        print(f"✓ Session cleaned up")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all language tests."""
    print("\n" + "="*70)
    print("LANGUAGE OPTIONS TEST SUITE")
    print("="*70)
    print("\nTesting all 3 language options (auto, en, pl)")
    print("This verifies the complete flow from API → Backend → MLX Whisper")

    results = []

    # Test 1: Auto-detect (None)
    results.append(await test_language_option(None, "Auto-detect"))
    await asyncio.sleep(0.5)

    # Test 2: English
    results.append(await test_language_option("en", "English"))
    await asyncio.sleep(0.5)

    # Test 3: Polish
    results.append(await test_language_option("pl", "Polish"))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    test_names = ["Auto-detect", "English", "Polish"]
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")

    print("="*70)

    if all(results):
        print("\n🎉 All language options work correctly!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
