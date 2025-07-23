# Enhanced Behavior Analysis with Pose Estimation

## Overview
The DuoPet AI behavior analysis system has been enhanced with SuperAnimal-Quadruped pose estimation, providing more accurate behavior classification through detailed keypoint tracking.

## Key Improvements

### 1. SuperAnimal-Quadruped Integration
- **39 keypoints** for detailed pose tracking (vs 20 in general models)
- Specifically designed for quadruped animals (dogs, cats)
- 294MB model with high accuracy

### 2. Enhanced Feature Extraction
- **Joint angles**: Neck, back curvature, leg positions
- **Movement velocities**: Track speed and direction of keypoints
- **Body symmetry**: Detect asymmetric movements indicating pain/injury
- **Temporal features**: Analyze patterns over time

### 3. Improved Behavior Classification
- Combines pose features with existing LSTM classifier
- Better detection of subtle behaviors
- More accurate abnormal behavior identification

## API Usage

### 1. Test Pose Estimation
```bash
curl -X POST "http://localhost:8000/api/v1/behavior/test-pose" \
  -F "image=@pet_photo.jpg"
```

Response:
```json
{
  "success": true,
  "data": {
    "num_keypoints": 39,
    "valid_keypoints": 32,
    "keypoint_names": ["nose", "upper_jaw", "lower_jaw", "right_eye", "left_eye"],
    "sample_keypoints": [
      {
        "name": "nose",
        "position": [245.3, 189.7],
        "confidence": 0.95
      }
    ]
  }
}
```

### 2. Analyze Video with Pose
```bash
curl -X POST "http://localhost:8000/api/v1/behavior/analyze" \
  -F "video=@pet_video.mp4" \
  -F "pet_type=dog" \
  -F "real_time=false"
```

Response includes pose usage information:
```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_abc123",
    "video_duration": 30.5,
    "behaviors": [...],
    "behavior_summary": {
      "walking": 15,
      "sitting": 8,
      "playing": 7
    },
    "pose_estimation_used": true,
    "pose_usage_percentage": 85.5
  }
}
```

### 3. Check Model Status
```bash
curl "http://localhost:8000/api/v1/behavior/model-status"
```

Response:
```json
{
  "success": true,
  "data": {
    "loaded_models": ["yolo_catdog", "lstm_dog", "lstm_cat"],
    "gpu_info": {
      "available": true,
      "device_name": "NVIDIA GeForce RTX 3060"
    },
    "pose_estimation": {
      "available": true,
      "model_type": "SuperAnimal-Quadruped",
      "keypoints": 39
    }
  }
}
```

## Testing

### 1. Run API Tests
```bash
cd D:\final_project\DuoPet_AI
conda activate duopet
python scripts/test_enhanced_behavior_api.py
```

### 2. Test Pose Estimation Only
```bash
python scripts/test_superanimal_adapter.py
```

### 3. Test with Sample Video
```bash
python scripts/test_behavior_api.py
```

## Performance Considerations

### Memory Usage
- SuperAnimal model: ~294MB on disk, ~500MB in memory
- TensorFlow session: ~1-2GB depending on batch size
- Recommended: 8GB+ RAM for smooth operation

### Processing Speed
- Pose extraction: ~100-200ms per frame
- With 5-frame sampling: ~6-12 FPS effective processing
- GPU acceleration recommended for real-time analysis

### Optimization Tips
1. Enable GPU if available (CUDA-compatible)
2. Process videos in background for files >10MB
3. Use frame sampling (every 5th frame by default)
4. Monitor memory with `/model-status` endpoint

## Troubleshooting

### TensorFlow Compatibility
If you encounter TF errors:
```bash
# Check TF version
python -c "import tensorflow as tf; print(tf.__version__)"
# Should be 2.14.0
```

### Model Loading Issues
1. Verify model files exist:
   ```bash
   ls models/behavior_analysis/pose_estimation/superanimal_quadruped/weights/
   ```

2. Check for corrupted downloads:
   - snapshot-700000.meta (~1MB)
   - snapshot-700000.data-00000-of-00001 (~294MB)
   - snapshot-700000.index

3. Re-download if needed:
   ```bash
   python scripts/download_superanimal_simple.py
   ```

### Memory Errors
If you get OOM errors:
1. Reset models: `POST /api/v1/behavior/reset-models`
2. Reduce batch size in configuration
3. Process smaller video segments

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Video Input   │────▶│  YOLO Detection  │────▶│ Pose Estimation │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ LSTM Classifier │◀────│ Feature Extract  │◀────│   39 Keypoints  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Behavior Result │
└─────────────────┘
```

## Future Enhancements

1. **Multi-pet tracking**: Track poses for multiple pets simultaneously
2. **Real-time streaming**: Process live video feeds
3. **Custom behaviors**: Train on specific behaviors for individual pets
4. **3D pose estimation**: Upgrade to 3D keypoint detection
5. **Mobile optimization**: Lightweight models for edge devices