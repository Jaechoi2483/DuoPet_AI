# SuperAnimal-Quadruped Pose Estimation Testing Guide

## Overview
This guide helps you test the newly integrated SuperAnimal-Quadruped pose estimation model in the DuoPet AI system.

## Prerequisites
1. Activate the conda environment:
   ```bash
   conda activate duopet
   ```

2. Ensure you're in the project directory:
   ```bash
   cd D:\final_project\DuoPet_AI
   ```

## Testing Steps

### 1. Test Model Loading
First, verify that TensorFlow can load the model:
```bash
python scripts/test_superanimal_loading.py
```

Expected output:
- TensorFlow version should be displayed
- Model should load without errors
- You should see placeholder and pose operations listed

### 2. Test the Adapter
Run the full adapter test:
```bash
python scripts/test_superanimal_adapter.py
```

This will:
- Load the SuperAnimal adapter
- Test on synthetic images
- Create visualizations in `test_output/` directory

### 3. Test with Real Pet Videos
If you have pet videos, you can test with:
```bash
python scripts/test_behavior_api.py
```

This uses the existing behavior analysis endpoint but now with pose estimation support.

## Integration Status

### Completed (Phase 1-2):
- ✅ Downloaded SuperAnimal-Quadruped model (294MB)
- ✅ Created model configuration
- ✅ Implemented SuperAnimalQuadrupedAdapter class
- ✅ Added TensorFlow 1.x/2.x compatibility
- ✅ Implemented keypoint extraction from heatmaps
- ✅ Added visualization methods

### Next Steps (Phase 3-5):
- 🔲 Implement enhanced feature extraction from keypoints
- 🔲 Integrate pose estimation into behavior analysis pipeline
- 🔲 Update API endpoints to use pose features
- 🔲 Add pose-based behavior classification
- 🔲 Performance optimization and caching

## Troubleshooting

### TensorFlow Version Issues
If you encounter TensorFlow errors:
```bash
# Check current TF version
python -c "import tensorflow as tf; print(tf.__version__)"

# Should be 2.14.0
```

### Model Loading Issues
If the model doesn't load:
1. Check model files exist:
   ```bash
   ls models/behavior_analysis/pose_estimation/superanimal_quadruped/weights/
   ```

2. Verify file sizes:
   - snapshot-700000.data-00000-of-00001 (~294MB)
   - snapshot-700000.index
   - snapshot-700000.meta

### Memory Issues
The model requires ~2GB of RAM. If you encounter memory errors:
1. Close other applications
2. Consider using a smaller batch size
3. Enable GPU if available

## Expected Results

When successfully running the adapter test, you should see:
1. Model loading confirmation
2. Detection of 39 keypoints for quadruped animals
3. Visualization images saved to `test_output/`
4. Confidence scores for each detected keypoint

## Sample Output Structure
```json
{
    "keypoints": [
        [x1, y1],  // nose
        [x2, y2],  // upper_jaw
        // ... 37 more keypoints
    ],
    "keypoint_names": ["nose", "upper_jaw", ...],
    "confidence_scores": [0.95, 0.87, ...],
    "valid_keypoints": [0, 1, 5, 6, ...]  // indices of keypoints above threshold
}
```

## Performance Notes
- First inference may be slow due to model initialization
- Subsequent predictions should be faster (~100-200ms per frame)
- GPU acceleration recommended for real-time processing