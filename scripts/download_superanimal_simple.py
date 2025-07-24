#!/usr/bin/env python3
"""
Simple SuperAnimal-Quadruped model downloader
"""

import os
from pathlib import Path
import requests
import zipfile
import yaml
from tqdm import tqdm

# Model information
MODEL_INFO = {
    "name": "SuperAnimal-Quadruped",
    "version": "1.0",
}

# Keypoint definitions for quadruped animals
KEYPOINT_NAMES = [
    "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
    "right_eye", "right_earbase", "right_earend", "right_antler_base", "right_antler_end",
    "left_eye", "left_earbase", "left_earend", "left_antler_base", "left_antler_end",
    "neck_base", "neck_end", "throat_base", "throat_end",
    "back_base", "back_end", "back_middle",
    "tail_base", "tail_end",
    "front_left_thigh", "front_left_knee", "front_left_paw",
    "front_right_thigh", "front_right_knee", "front_right_paw",
    "back_left_thigh", "back_left_knee", "back_left_paw",
    "back_right_thigh", "back_right_knee", "back_right_paw",
    "spine1", "spine2", "spine3"
]

def download_from_dlclibrary():
    """Download using dlclibrary"""
    try:
        from dlclibrary import download_huggingface_model
        
        # Get project root
        project_root = Path(__file__).parent.parent
        base_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
        base_path.mkdir(parents=True, exist_ok=True)
        
        model_dir = base_path / "weights"
        model_dir.mkdir(exist_ok=True)
        
        print("Downloading SuperAnimal-Quadruped model using dlclibrary...")
        download_huggingface_model("superanimal_quadruped", str(model_dir))
        
        # Create our config
        config_dir = base_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        config = {
            "model_name": MODEL_INFO["name"],
            "version": MODEL_INFO["version"],
            "keypoints": {
                "names": KEYPOINT_NAMES,
                "count": len(KEYPOINT_NAMES),
                "skeleton": [
                    # Head connections
                    ["nose", "upper_jaw"], ["upper_jaw", "lower_jaw"],
                    ["nose", "right_eye"], ["nose", "left_eye"],
                    ["right_eye", "right_earbase"], ["left_eye", "left_earbase"],
                    
                    # Neck and back
                    ["throat_base", "neck_base"], ["neck_base", "back_base"],
                    ["back_base", "back_middle"], ["back_middle", "back_end"],
                    ["back_end", "tail_base"], ["tail_base", "tail_end"],
                    
                    # Front legs
                    ["neck_base", "front_left_thigh"], ["front_left_thigh", "front_left_knee"],
                    ["front_left_knee", "front_left_paw"],
                    ["neck_base", "front_right_thigh"], ["front_right_thigh", "front_right_knee"],
                    ["front_right_knee", "front_right_paw"],
                    
                    # Back legs
                    ["back_end", "back_left_thigh"], ["back_left_thigh", "back_left_knee"],
                    ["back_left_knee", "back_left_paw"],
                    ["back_end", "back_right_thigh"], ["back_right_thigh", "back_right_knee"],
                    ["back_right_knee", "back_right_paw"],
                    
                    # Spine
                    ["spine1", "spine2"], ["spine2", "spine3"]
                ]
            },
            "input_size": [640, 480],
            "confidence_threshold": 0.5,
            "batch_size": 8,
            "cache_size": 100
        }
        
        config_path = config_dir / "model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        print(f"Created model configuration at {config_path}")
        print("Model downloaded successfully!")
        print(f"Model location: {base_path}")
        
        # Check downloaded files
        weights_dir = base_path / "weights"
        if weights_dir.exists():
            files = list(weights_dir.glob("*"))
            print(f"\nDownloaded files ({len(files)}):")
            for file in files[:5]:  # Show first 5 files
                print(f"  - {file.name}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")
        
        return True
        
    except ImportError:
        print("dlclibrary not installed. Please install it first:")
        print("pip install dlclibrary")
        return False
    except Exception as e:
        print(f"Failed to download model: {str(e)}")
        return False

def main():
    """Main download function"""
    print("Starting SuperAnimal-Quadruped model download...")
    
    if download_from_dlclibrary():
        print("\n✅ Model setup completed successfully!")
    else:
        print("\n❌ Failed to download model")

if __name__ == "__main__":
    main()