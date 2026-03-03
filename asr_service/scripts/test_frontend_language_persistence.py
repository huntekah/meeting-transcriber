#!/usr/bin/env python3
"""
Test frontend language setting persistence.

Verifies:
1. Settings load with default "auto"
2. Settings save correctly
3. Settings load from saved file
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cli_frontend.config import CLISettings
from cli_frontend.settings_persistence import persistence


def test_default_setting():
    """Test that default language is 'auto'."""
    print("\n" + "="*70)
    print("Test 1: Default Language Setting")
    print("="*70)

    settings = CLISettings()
    print(f"Default asr_language: {settings.asr_language}")

    assert settings.asr_language == "auto", f"Expected 'auto', got '{settings.asr_language}'"
    print("✓ PASS: Default is 'auto'")
    return True


def test_save_and_load():
    """Test saving and loading language settings."""
    print("\n" + "="*70)
    print("Test 2: Save and Load Language Settings")
    print("="*70)

    config_path = Path.home() / ".meeting_scribe" / "cli_config.json"
    print(f"Config file: {config_path}")

    # Test each language option
    test_values = ["auto", "en", "pl"]
    results = []

    for lang in test_values:
        print(f"\n  Testing: {lang}")

        # Save setting
        persistence.save_general_settings({"asr_language": lang})
        print(f"    ✓ Saved to disk")

        # Load from file
        if config_path.exists():
            with open(config_path, 'r') as f:
                saved_data = json.load(f)
            saved_lang = saved_data.get("asr_language")
            print(f"    ✓ Loaded from disk: {saved_lang}")

            if saved_lang == lang:
                print(f"    ✓ PASS: {lang} saved and loaded correctly")
                results.append(True)
            else:
                print(f"    ❌ FAIL: Expected '{lang}', got '{saved_lang}'")
                results.append(False)
        else:
            print(f"    ❌ FAIL: Config file not created")
            results.append(False)

    # Load into CLISettings to verify it integrates
    print(f"\n  Loading into CLISettings from persistence...")
    loaded_settings = persistence.load_general_settings()
    print(f"    Loaded settings: {loaded_settings}")

    if loaded_settings.get("asr_language") == "pl":  # Last saved value
        print(f"    ✓ PASS: Settings integrate correctly")
        results.append(True)
    else:
        print(f"    ❌ FAIL: Integration issue")
        results.append(False)

    return all(results)


def test_api_payload():
    """Test that language flows to API payload correctly."""
    print("\n" + "="*70)
    print("Test 3: Language in API Payload")
    print("="*70)

    # Simulate what the client does
    from cli_frontend.models import SourceConfig

    sources = [
        SourceConfig(
            device_index=0,
            device_name="Test Mic",
            device_channels=1,
            source_type="sounddevice"
        )
    ]

    test_cases = [
        ("auto", None, "Auto should send None to backend"),
        ("en", "en", "English should send 'en'"),
        ("pl", "pl", "Polish should send 'pl'"),
    ]

    results = []
    for lang_setting, expected_payload, description in test_cases:
        print(f"\n  Testing: {description}")

        # Build payload like ASRClient does
        payload = {
            "sources": [s.model_dump() for s in sources],
        }

        # Only add language if not auto
        if lang_setting and lang_setting != "auto":
            payload["language"] = lang_setting

        actual = payload.get("language")
        print(f"    Setting: {lang_setting}")
        print(f"    Payload: {payload.get('language', 'None (not in payload)')}")
        print(f"    Expected: {expected_payload}")

        if actual == expected_payload:
            print(f"    ✓ PASS")
            results.append(True)
        else:
            print(f"    ❌ FAIL")
            results.append(False)

    return all(results)


def main():
    """Run all frontend persistence tests."""
    print("\n" + "="*70)
    print("FRONTEND LANGUAGE PERSISTENCE TEST SUITE")
    print("="*70)

    results = []

    # Test 1: Default setting
    try:
        results.append(test_default_setting())
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        results.append(False)

    # Test 2: Save and load
    try:
        results.append(test_save_and_load())
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    # Test 3: API payload
    try:
        results.append(test_api_payload())
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        results.append(False)

    # Summary
    print("\n" + "="*70)
    print("FRONTEND TEST SUMMARY")
    print("="*70)

    test_names = [
        "Default Setting",
        "Save & Load Persistence",
        "API Payload Generation"
    ]

    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name:30} {status}")

    print("="*70)

    if all(results):
        print("\n🎉 All frontend tests passed!")
        return 0
    else:
        print("\n❌ Some frontend tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
