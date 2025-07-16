"""
Simple script to check if model files exist
"""

import os

def check_model_files():
    """Check if all model files are properly copied"""
    
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_path = os.path.join(base_path, 'models', 'health_diagnosis')
    
    print("🔍 Checking model files...")
    print(f"Base path: {models_path}")
    print("-" * 50)
    
    # Check BCS model
    bcs_model = os.path.join(models_path, 'bcs', 'bcs_efficientnet_v1.h5')
    if os.path.exists(bcs_model):
        size_mb = os.path.getsize(bcs_model) / (1024 * 1024)
        print(f"✅ BCS model found: {size_mb:.1f} MB")
    else:
        print("❌ BCS model NOT found")
        
    # Check skin disease models
    skin_path = os.path.join(models_path, 'skin_disease')
    
    # Classification models
    class_models = {
        'cat_binary': 'classification/cat_binary/checkpoint',
        'dog_binary': 'classification/dog_binary/checkpoint',
        'dog_multi_136': 'classification/dog_multi_136/checkpoint',
        'dog_multi_456': 'classification/dog_multi_456/checkpoint'
    }
    
    print("\n📁 Skin Disease Classification Models:")
    for name, path in class_models.items():
        full_path = os.path.join(skin_path, path)
        if os.path.exists(full_path):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
            
    # Segmentation models
    print("\n📁 Skin Disease Segmentation Models:")
    seg_models = ['cat_A2', 'dog_A1', 'dog_A2', 'dog_A3', 'dog_A4', 'dog_A5', 'dog_A6']
    for model in seg_models:
        checkpoint = os.path.join(skin_path, 'segmentation', model, 'checkpoint')
        if os.path.exists(checkpoint):
            print(f"  ✅ {model}")
        else:
            print(f"  ❌ {model}")
            
    # Check behavior models
    print("\n📁 Behavior Analysis Models:")
    behavior_path = os.path.join(base_path, 'models', 'behavior_analysis')
    
    behavior_models = {
        'YOLO CatDog': 'detection/behavior_yolo_catdog_v1.pt',
        'YOLO Base': 'detection/behavior_yolo_base_v1.pt',
        'Cat LSTM': 'classification/behavior_cat_lstm_v1.pth',
        'Dog LSTM': 'classification/behavior_dog_lstm_v1.pth'
    }
    
    for name, path in behavior_models.items():
        full_path = os.path.join(behavior_path, path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {name}")
            
    # Check chatbot models
    print("\n📁 Chatbot Models:")
    chatbot_path = os.path.join(base_path, 'models', 'chatbot')
    
    chatbot_models = {
        'KoGPT2': 'chatbot_kogpt2_v1/pytorch_model.bin',
        'RoBERTa': 'chatbot_roberta_v1/pytorch_model.bin'
    }
    
    for name, path in chatbot_models.items():
        full_path = os.path.join(chatbot_path, path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✅ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {name}")

if __name__ == "__main__":
    check_model_files()