"""
Test script for BCS model
"""

import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.health_diagnosis.predict import predict_bcs
from common.logger import get_logger

logger = get_logger(__name__)


def create_dummy_images(num_images=13):
    """Create dummy images for testing"""
    images = []
    for i in range(num_images):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


def test_bcs_prediction():
    """Test BCS prediction"""
    logger.info("Starting BCS model test...")
    
    try:
        # Create dummy images
        test_images = create_dummy_images(13)
        logger.info(f"Created {len(test_images)} test images")
        
        # Test prediction
        logger.info("Testing BCS prediction...")
        result = predict_bcs(test_images)
        
        logger.info("BCS Prediction Result:")
        logger.info(f"- BCS Score: {result['bcs_score']}")
        logger.info(f"- Category: {result['category']}")
        logger.info(f"- Confidence: {result['confidence']:.2f}")
        logger.info(f"- Recommendations: {result['recommendations']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_bcs_prediction()
    if success:
        print("✅ BCS model test passed!")
    else:
        print("❌ BCS model test failed!")