#!/usr/bin/env python3
"""
Test script for SuperAnimal-Quadruped Adapter
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.behavior_analysis.superanimal_adapter import SuperAnimalQuadrupedAdapter
from common.logger import get_logger

logger = get_logger(__name__)

def test_adapter_loading():
    """Test if the adapter loads correctly"""
    logger.info("Testing SuperAnimalQuadrupedAdapter loading...")
    
    try:
        adapter = SuperAnimalQuadrupedAdapter()
        logger.info("✓ Adapter loaded successfully")
        
        # Check configuration
        logger.info(f"Number of keypoints: {adapter.num_keypoints}")
        logger.info(f"Input size: {adapter.input_size}")
        logger.info(f"Confidence threshold: {adapter.confidence_threshold}")
        
        # Show first 5 keypoint names
        logger.info(f"Keypoint names (first 5): {adapter.keypoint_names[:5]}")
        
        return adapter
        
    except Exception as e:
        logger.error(f"✗ Failed to load adapter: {str(e)}")
        return None

def test_on_sample_image(adapter, image_path=None):
    """Test the adapter on a sample image"""
    logger.info("\nTesting on sample image...")
    
    if image_path is None:
        # Create a simple test image (300x300 with a rectangle)
        image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        # Draw a rectangle to simulate a pet
        cv2.rectangle(image, (50, 50), (250, 250), (100, 100, 100), -1)
        # Add some features
        cv2.circle(image, (100, 100), 20, (50, 50, 50), -1)  # Eye
        cv2.circle(image, (200, 100), 20, (50, 50, 50), -1)  # Eye
        image_path = "test_image"
    else:
        # Load real image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return
    
    try:
        # Test with full image
        logger.info("Testing with full image...")
        result = adapter.predict(image)
        
        logger.info(f"✓ Prediction completed")
        logger.info(f"Valid keypoints detected: {len(result['valid_keypoints'])}")
        
        if result['valid_keypoints']:
            # Show some detected keypoints
            for idx in result['valid_keypoints'][:3]:
                kp_name = result['keypoint_names'][idx]
                kp_pos = result['keypoints'][idx]
                conf = result['confidence_scores'][idx]
                logger.info(f"  - {kp_name}: ({kp_pos[0]:.1f}, {kp_pos[1]:.1f}) [conf: {conf:.3f}]")
        
        # Test with bounding box
        h, w = image.shape[:2]
        bbox = [w*0.2, h*0.2, w*0.8, h*0.8]
        logger.info(f"\nTesting with bounding box: {bbox}")
        
        result_bbox = adapter.predict(image, bbox)
        logger.info(f"✓ Prediction with bbox completed")
        logger.info(f"Valid keypoints detected: {len(result_bbox['valid_keypoints'])}")
        
        # Visualize results
        if result['valid_keypoints']:
            vis_image = adapter.visualize_keypoints(image, result)
            
            # Save visualization
            output_path = project_root / "test_output" / "superanimal_test_result.jpg"
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"✓ Visualization saved to: {output_path}")
            
            # Display if not running in headless mode
            try:
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
                plt.title(f"SuperAnimal Keypoints ({len(result['valid_keypoints'])} detected)")
                plt.axis('off')
                plt.tight_layout()
                
                # Save plot
                plot_path = project_root / "test_output" / "superanimal_test_plot.png"
                plt.savefig(str(plot_path))
                logger.info(f"✓ Plot saved to: {plot_path}")
                plt.close()
            except Exception as e:
                logger.warning(f"Could not display/save plot: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_on_pet_images(adapter):
    """Test on actual pet images if available"""
    logger.info("\nLooking for pet images to test...")
    
    # Check for sample images
    sample_dirs = [
        project_root / "data" / "test_images",
        project_root / "test_data" / "images",
        project_root / "samples"
    ]
    
    test_images = []
    for sample_dir in sample_dirs:
        if sample_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                test_images.extend(list(sample_dir.glob(ext)))
    
    if not test_images:
        logger.info("No test images found. Creating sample dog/cat images...")
        # Create simple synthetic pet images
        create_sample_pet_images()
        test_images = list((project_root / "test_output").glob("sample_*.jpg"))
    
    # Test on each image
    for img_path in test_images[:3]:  # Test first 3 images
        logger.info(f"\nTesting on: {img_path.name}")
        result = test_on_sample_image(adapter, str(img_path))
        
        if result:
            logger.info(f"  Detection summary: {len(result['valid_keypoints'])} keypoints")

def create_sample_pet_images():
    """Create simple synthetic pet images for testing"""
    output_dir = project_root / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Create dog-like image
    dog_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Body
    cv2.ellipse(dog_img, (300, 250), (150, 80), 0, 0, 360, (139, 90, 43), -1)
    # Head
    cv2.circle(dog_img, (150, 200), 60, (160, 110, 60), -1)
    # Eyes
    cv2.circle(dog_img, (130, 180), 10, (0, 0, 0), -1)
    cv2.circle(dog_img, (170, 180), 10, (0, 0, 0), -1)
    # Nose
    cv2.circle(dog_img, (150, 210), 8, (0, 0, 0), -1)
    # Legs
    for x in [200, 250, 350, 400]:
        cv2.rectangle(dog_img, (x-15, 280), (x+15, 350), (139, 90, 43), -1)
    
    cv2.imwrite(str(output_dir / "sample_dog.jpg"), dog_img)
    
    # Create cat-like image
    cat_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Body
    cv2.ellipse(cat_img, (350, 250), (120, 70), 0, 0, 360, (80, 80, 80), -1)
    # Head
    cv2.circle(cat_img, (200, 200), 50, (100, 100, 100), -1)
    # Ears (triangles)
    pts1 = np.array([[170, 170], [160, 140], [190, 160]], np.int32)
    pts2 = np.array([[210, 160], [240, 140], [230, 170]], np.int32)
    cv2.fillPoly(cat_img, [pts1], (100, 100, 100))
    cv2.fillPoly(cat_img, [pts2], (100, 100, 100))
    # Eyes
    cv2.ellipse(cat_img, (185, 190), (8, 12), 0, 0, 360, (0, 150, 0), -1)
    cv2.ellipse(cat_img, (215, 190), (8, 12), 0, 0, 360, (0, 150, 0), -1)
    # Nose
    pts_nose = np.array([[200, 210], [195, 215], [205, 215]], np.int32)
    cv2.fillPoly(cat_img, [pts_nose], (200, 100, 100))
    
    cv2.imwrite(str(output_dir / "sample_cat.jpg"), cat_img)
    
    logger.info("Created sample pet images")

def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("SuperAnimal-Quadruped Adapter Test")
    logger.info("=" * 60)
    
    # Test 1: Load adapter
    adapter = test_adapter_loading()
    if adapter is None:
        logger.error("Failed to load adapter. Exiting.")
        return
    
    # Test 2: Test on synthetic image
    logger.info("\n" + "-" * 40)
    test_on_sample_image(adapter)
    
    # Test 3: Test on pet images
    logger.info("\n" + "-" * 40)
    test_on_pet_images(adapter)
    
    logger.info("\n" + "=" * 60)
    logger.info("Test completed!")

if __name__ == "__main__":
    main()