"""
눈 질환 모델 구조 분석
"""
import tensorflow as tf
import h5py
from pathlib import Path

# TensorFlow eager execution 활성화
tf.config.run_functions_eagerly(True)

def analyze_eye_model():
    model_path = Path("models/health_diagnosis/eye_disease/eye_disease_fixed.h5")
    
    print(f"\n🔍 눈 질환 모델 분석: {model_path}")
    print("="*60)
    
    # 1. H5 파일 구조 확인
    with h5py.File(model_path, 'r') as f:
        print("\n1️⃣ H5 파일 구조:")
        
        def print_structure(name, obj, indent=0):
            if indent < 3:  # 너무 깊이 들어가지 않도록
                print(" " * indent + name)
                if hasattr(obj, 'keys'):
                    for key in obj.keys():
                        print_structure(f"{name}/{key}", obj[key], indent + 2)
        
        for key in f.keys():
            print_structure(key, f[key])
    
    # 2. 모델 로드 시도
    try:
        print("\n2️⃣ 모델 로드 시도...")
        
        # Custom objects 정의
        custom_objects = {
            'Functional': tf.keras.models.Model,
            'TFOpLambda': tf.keras.layers.Lambda
        }
        
        # compile=False로 로드
        model = tf.keras.models.load_model(
            str(model_path), 
            custom_objects=custom_objects,
            compile=False
        )
        
        print("✅ 모델 로드 성공!")
        
        # 3. 모델 구조 확인
        print(f"\n3️⃣ 모델 구조:")
        print(f"   - 입력 shape: {model.input_shape}")
        print(f"   - 출력 shape: {model.output_shape}")
        print(f"   - 전체 레이어 수: {len(model.layers)}")
        
        # 레이어 타입 분석
        layer_types = {}
        for layer in model.layers:
            layer_type = layer.__class__.__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        print(f"\n4️⃣ 레이어 타입별 통계:")
        for layer_type, count in sorted(layer_types.items()):
            print(f"   - {layer_type}: {count}개")
        
        # 처음 10개 레이어
        print(f"\n5️⃣ 처음 10개 레이어:")
        for i, layer in enumerate(model.layers[:10]):
            print(f"   {i}: {layer.__class__.__name__} - {layer.name}")
        
        # 마지막 5개 레이어
        print(f"\n6️⃣ 마지막 5개 레이어:")
        for i, layer in enumerate(model.layers[-5:], len(model.layers)-5):
            print(f"   {i}: {layer.__class__.__name__} - {layer.name}")
        
        # TFOpLambda 레이어 확인
        tfoplambda_layers = [l for l in model.layers if l.__class__.__name__ == 'TFOpLambda']
        if tfoplambda_layers:
            print(f"\n⚠️ TFOpLambda 레이어 발견: {len(tfoplambda_layers)}개")
            for layer in tfoplambda_layers[:5]:  # 처음 5개만
                print(f"   - {layer.name}")
        
    except Exception as e:
        print(f"\n❌ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_eye_model()