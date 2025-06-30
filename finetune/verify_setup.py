#!/usr/bin/env python3
"""Verify the finetuning setup is correct."""

import os
import sys
import yaml
import json
from pathlib import Path

def print_info(msg):
    print(f"\033[1;34m[INFO]\033[0m {msg}")

def print_success(msg):
    print(f"\033[1;32m[SUCCESS]\033[0m {msg}")

def print_error(msg):
    print(f"\033[1;31m[ERROR]\033[0m {msg}")

def verify_environment():
    """Verify Python environment and dependencies."""
    print_info("Checking Python environment...")
    
    required_packages = [
        'torch', 'torchaudio', 'soundfile', 'numpy', 
        'google.cloud.texttospeech', 'google.genai', 
        'audiomentations', 'tqdm', 'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'google.cloud.texttospeech':
                import google.cloud.texttospeech
            elif package == 'google.genai':
                import google.genai
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
        return False
    
    print_success("All required packages installed")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_info("CUDA not available - will use CPU (slower)")
    except:
        pass
    
    return True

def verify_api_keys():
    """Check if API keys are set."""
    print_info("Checking API keys...")
    
    issues = []
    
    if not os.environ.get('GOOGLE_API_KEY'):
        issues.append("GOOGLE_API_KEY not set")
    
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        if not os.path.exists(os.path.expanduser('~/.config/gcloud/application_default_credentials.json')):
            issues.append("Google Cloud credentials not found")
    
    if issues:
        for issue in issues:
            print_error(issue)
        return False
    
    print_success("API keys configured")
    return True

def verify_pretrained_models():
    """Check if pretrained models exist."""
    print_info("Checking pretrained models...")
    
    models = {
        '8K': 'pretrained_models/MossFormer2_SS_8K.pth',
        '16K': 'pretrained_models/MossFormer2_SS_16K.pth'
    }
    
    found = []
    for name, path in models.items():
        if os.path.exists(path):
            found.append(name)
            print_success(f"Found {name} model: {path}")
        else:
            print_info(f"{name} model not found: {path}")
    
    if not found:
        print_error("No pretrained models found!")
        return False
    
    return True

def verify_directory_structure():
    """Check directory structure."""
    print_info("Checking directory structure...")
    
    required_dirs = [
        'train/speech_separation',
        'config/train',
        'finetune'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    if missing:
        print_error(f"Missing directories: {', '.join(missing)}")
        return False
    
    print_success("Directory structure OK")
    return True

def verify_config_templates():
    """Verify config templates are valid."""
    print_info("Checking config templates...")
    
    templates = [
        'finetune/config_template_8000.yaml',
        'finetune/config_template_16000.yaml'
    ]
    
    for template in templates:
        if not os.path.exists(template):
            print_error(f"Config template not found: {template}")
            return False
        
        try:
            with open(template, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['network', 'sampling_rate', 'tr_list', 'cv_list', 'tt_list']
            missing_fields = [f for f in required_fields if f not in config]
            
            if missing_fields:
                print_error(f"Missing fields in {template}: {', '.join(missing_fields)}")
                return False
            
            print_success(f"Config template valid: {template}")
        except Exception as e:
            print_error(f"Error parsing {template}: {e}")
            return False
    
    return True

def main():
    """Run all verification checks."""
    print("=== ClearerVoice Finetuning Setup Verification ===\n")
    
    checks = [
        ("Environment", verify_environment),
        ("API Keys", verify_api_keys),
        ("Pretrained Models", verify_pretrained_models),
        ("Directory Structure", verify_directory_structure),
        ("Config Templates", verify_config_templates)
    ]
    
    all_passed = True
    results = []
    
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results.append((name, False))
            all_passed = False
        print()
    
    print("=== Summary ===")
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    print()
    if all_passed:
        print_success("All checks passed! Ready to run training.")
        print("\nNext step: bash finetune/run_training.sh <num_conversations> <sample_rate>")
    else:
        print_error("Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()