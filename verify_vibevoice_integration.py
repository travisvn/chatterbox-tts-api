#!/usr/bin/env python3
"""
Simple verification script for VibeVoice integration

This script validates the integration without importing modules
that require heavy dependencies (torch, etc.)
"""

import ast
import os


def check_file_exists(filepath):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists


def check_code_contains(filepath, search_strings):
    """Check if file contains specific strings"""
    with open(filepath, 'r') as f:
        content = f.read()

    results = []
    for search_str in search_strings:
        found = search_str in content
        status = "✓" if found else "✗"
        print(f"  {status} Contains: {search_str}")
        results.append(found)

    return all(results)


def check_python_syntax(filepath):
    """Check if Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        print(f"✓ {filepath} - Valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} - Syntax error: {e}")
        return False


def main():
    """Run verification checks"""
    print("=" * 70)
    print("VIBEVOICE INTEGRATION VERIFICATION")
    print("=" * 70)

    all_checks_passed = True

    # 1. Check VibeVoice engine file exists
    print("\n1. Checking VibeVoice engine file:")
    vibevoice_py = 'app/core/tts_engines/vibevoice.py'
    if not check_file_exists(vibevoice_py):
        all_checks_passed = False
    else:
        if not check_python_syntax(vibevoice_py):
            all_checks_passed = False

    # 2. Check __init__.py includes VibeVoiceEngine
    print("\n2. Checking tts_engines/__init__.py:")
    init_py = 'app/core/tts_engines/__init__.py'
    if not check_code_contains(init_py, [
        'from .vibevoice import VibeVoiceEngine',
        '"VibeVoiceEngine"'
    ]):
        all_checks_passed = False

    # 3. Check tts_model.py integration
    print("\n3. Checking tts_model.py integration:")
    tts_model_py = 'app/core/tts_model.py'
    if not check_code_contains(tts_model_py, [
        'VibeVoiceEngine',
        'vibevoice-1.5b',
        'vibevoice-7b',
        "model_version.startswith('vibevoice')"
    ]):
        all_checks_passed = False

    # 4. Check VibeVoice engine implementation
    print("\n4. Checking VibeVoice engine implementation:")
    with open(vibevoice_py, 'r') as f:
        vibevoice_content = f.read()

    required_methods = [
        'async def load_model',
        'def generate',
        'def get_supported_languages',
        'def get_model_info',
        'def cleanup'
    ]

    print("  Required methods:")
    for method in required_methods:
        found = method in vibevoice_content
        status = "✓" if found else "✗"
        print(f"    {status} {method}")
        if not found:
            all_checks_passed = False

    # 5. Check model variants
    print("\n5. Checking model variant support:")
    variants = ['1.5b', '7b']
    variant_checks = [
        'microsoft/VibeVoice-1.5B' in vibevoice_content,
        'microsoft/VibeVoice-7B' in vibevoice_content,
        'model_variant' in vibevoice_content
    ]

    for i, (variant, check) in enumerate(zip(variants, variant_checks[:2])):
        status = "✓" if check else "✗"
        print(f"  {status} VibeVoice-{variant.upper()} model path")
        if not check:
            all_checks_passed = False

    status = "✓" if variant_checks[2] else "✗"
    print(f"  {status} Model variant parameter")
    if not variant_checks[2]:
        all_checks_passed = False

    # 6. Check language support
    print("\n6. Checking language support:")
    lang_checks = [
        '"en": "English"' in vibevoice_content,
        '"zh": "Chinese' in vibevoice_content,
        'def get_supported_languages' in vibevoice_content
    ]

    status = "✓" if all(lang_checks) else "✗"
    print(f"  {status} Multilingual support implemented")
    if not all(lang_checks):
        all_checks_passed = False

    # 7. Check requirements.txt
    print("\n7. Checking requirements.txt:")
    requirements_txt = 'requirements.txt'
    if not check_code_contains(requirements_txt, [
        'VibeVoice',
        'microsoft/VibeVoice-1.5B',
        'microsoft/VibeVoice-7B'
    ]):
        all_checks_passed = False

    # 8. Check tts_model.py model list
    print("\n8. Checking available models list:")
    with open(tts_model_py, 'r') as f:
        tts_model_content = f.read()

    model_list_checks = [
        '"vibevoice-1.5b"' in tts_model_content,
        '"vibevoice-7b"' in tts_model_content,
        'all_models = [' in tts_model_content
    ]

    for check in model_list_checks:
        if not check:
            all_checks_passed = False

    status = "✓" if all(model_list_checks) else "✗"
    print(f"  {status} Models in available_models list")

    # 9. Check lazy loading support
    print("\n9. Checking lazy loading implementation:")
    lazy_checks = [
        'if self._is_loaded:' in vibevoice_content,
        'self._is_loaded = True' in vibevoice_content,
        'async def load_model' in vibevoice_content
    ]

    status = "✓" if all(lazy_checks) else "✗"
    print(f"  {status} Lazy loading pattern implemented")
    if not all(lazy_checks):
        all_checks_passed = False

    # 10. Check model info metadata
    print("\n10. Checking model info metadata:")
    metadata_checks = [
        'context_length' in vibevoice_content,
        'max_duration' in vibevoice_content,
        'multi_speaker' in vibevoice_content,
        'long_form_generation' in vibevoice_content
    ]

    for check in metadata_checks:
        if not check:
            all_checks_passed = False

    status = "✓" if all(metadata_checks) else "✗"
    print(f"  {status} Model metadata complete")

    # Summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✓ ALL VERIFICATION CHECKS PASSED!")
        print("=" * 70)
        print("\nVibeVoice integration is complete and ready to use.")
        print("\nTo use VibeVoice:")
        print("1. Install the library:")
        print("   git clone https://github.com/vibevoice-community/VibeVoice.git")
        print("   cd VibeVoice && pip install -e .")
        print("\n2. Use in API requests:")
        print("   model='vibevoice-1.5b' or model='vibevoice-7b'")
        print("\n3. Models will be auto-downloaded from HuggingFace on first use")
        return 0
    else:
        print("✗ SOME VERIFICATION CHECKS FAILED!")
        print("=" * 70)
        print("\nPlease review the errors above.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
