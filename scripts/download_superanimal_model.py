#!/usr/bin/env python3
"""
Download SuperAnimal-Quadruped model from DeepLabCut Model Zoo
"""

import os
import sys
from pathlib import Path
import requests
import zipfile
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.logger import get_logger

logger = get_logger(__name__)

# Model information
MODEL_INFO = {
    "name": "SuperAnimal-Quadruped",
    "version": "1.0",
    "huggingface_url": "https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
    "files": {
        "model_weights": "https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/resolve/main/DLC_quadruped_model.pth",
        "config": "https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/resolve/main/config.yaml",
        "pose_cfg": "https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped/resolve/main/pose_cfg.yaml"
    }
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

def download_file(url: str, dest_path: Path, description: str):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=description) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
        logger.info(f"Downloaded {description} to {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {description}: {str(e)}")
        return False

def setup_model_directory():
    """Create directory structure for SuperAnimal model"""
    base_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (base_path / "weights").mkdir(exist_ok=True)
    (base_path / "config").mkdir(exist_ok=True)
    
    return base_path

def create_model_config(base_path: Path):
    """Create configuration file for the model"""
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
    
    config_path = base_path / "config" / "model_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    logger.info(f"Created model configuration at {config_path}")
    return config_path

def download_from_dlclibrary():
    """Alternative: Download using dlclibrary"""
    try:
        from dlclibrary import download_huggingface_model
        
        base_path = setup_model_directory()
        model_dir = base_path / "weights"
        
        logger.info("Downloading SuperAnimal-Quadruped model using dlclibrary...")
        download_huggingface_model("superanimal_quadruped", model_dir)
        
        # Create our config
        create_model_config(base_path)
        
        logger.info("Model downloaded successfully!")
        return True
        
    except ImportError:
        logger.warning("dlclibrary not installed. Please install it first:")
        logger.warning("pip install deeplabcut[dlclibrary]")
        return False
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False

def verify_model_files(base_path: Path):
    """Verify that all necessary model files exist"""
    required_files = [
        base_path / "weights" / "snapshot-1030000.pb",  # Model weights
        base_path / "config" / "model_config.yaml"       # Our config
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            logger.info(f"✓ Found: {file_path}")
        else:
            logger.error(f"✗ Missing: {file_path}")
            all_exist = False
            
    return all_exist

def main():
    """Main download function"""
    logger.info("Starting SuperAnimal-Quadruped model download...")
    
    # Try using dlclibrary first
    if download_from_dlclibrary():
        base_path = project_root / "models" / "behavior_analysis" / "pose_estimation" / "superanimal_quadruped"
        if verify_model_files(base_path):
            logger.info("✅ Model setup completed successfully!")
            logger.info(f"Model location: {base_path}")
        else:
            logger.error("❌ Some model files are missing!")
            sys.exit(1)
    else:
        logger.error("❌ Failed to download model")
        logger.info("Please install dlclibrary: pip install deeplabcut[dlclibrary]")
        sys.exit(1)

if __name__ == "__main__":
    main()