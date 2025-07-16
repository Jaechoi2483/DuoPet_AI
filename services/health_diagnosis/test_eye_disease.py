"""
Test script for Eye Disease model
"""

import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.health_diagnosis.predict import predict_eye_disease
from common.logger import get_logger

logger = get_logger(__name__)


def create_dummy_eye_image():
    """Create a dummy eye image for testing"""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img


def test_eye_disease_prediction():
    """Test eye disease prediction"""
    logger.info("Starting eye disease model test...")
    
    try:
        # Create dummy image
        test_image = create_dummy_eye_image()
        logger.info("Created test eye image")
        
        # Test prediction
        logger.info("Testing eye disease prediction...")
        result = predict_eye_disease(test_image)
        
        logger.info("Eye Disease Prediction Result:")
        logger.info(f"- Disease Detected: {result['disease_detected']}")
        logger.info(f"- Disease Type: {result['disease_type']}")
        logger.info(f"- Confidence: {result['confidence']:.2f}")
        logger.info(f"- Severity: {result['severity']}")
        logger.info(f"- Recommendations: {result['recommendations']}")
        
        if 'all_predictions' in result:
            logger.info("All predictions:")
            for disease, prob in result['all_predictions'].items():
                logger.info(f"  - {disease}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_eye_disease_prediction()
    if success:
        print("✅ Eye disease model test completed!")
    else:
        print("❌ Eye disease model test failed!")