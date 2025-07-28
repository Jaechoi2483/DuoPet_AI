"""
원본 피부 질환 모델의 정확한 구조 분석
"""
import h5py
import json
import sys
from pathlib import Path

def analyze_h5_model(model_path):
    """H5 모델 파일의 상세 구조 분석"""
    print(f"\n🔍 모델 분석: {model_path.name}")
    print("="*60)
    
    with h5py.File(model_path, 'r') as f:
        # 모델 설정 정보
        if 'model_config' in f.attrs:
            config = json.loads(f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config'])
            print("\n📋 모델 설정:")
            print(f"  - 클래스: {config.get('class_name', 'Unknown')}")
            print(f"  - 백엔드: {config.get('backend', 'Unknown')}")
            print(f"  - Keras 버전: {config.get('keras_version', 'Unknown')}")
            
            # 레이어 구조 분석
            if 'config' in config and 'layers' in config['config']:
                layers = config['config']['layers']
                print(f"\n📊 레이어 구조: 총 {len(layers)}개 레이어")
                
                # 처음 10개와 마지막 5개 레이어만 표시
                print("\n  [처음 10개 레이어]")
                for i, layer in enumerate(layers[:10]):
                    layer_config = layer.get('config', {})
                    print(f"  {i}: {layer['class_name']} - {layer_config.get('name', 'unnamed')}")
                
                if len(layers) > 15:
                    print("\n  ... 중간 레이어 생략 ...\n")
                    
                print("\n  [마지막 5개 레이어]")
                for i, layer in enumerate(layers[-5:], len(layers)-5):
                    layer_config = layer.get('config', {})
                    print(f"  {i}: {layer['class_name']} - {layer_config.get('name', 'unnamed')}")
        
        # 가중치 구조 분석
        if 'model_weights' in f:
            print("\n🔧 가중치 구조:")
            weights_group = f['model_weights']
            
            # 레이어별 가중치 정리
            layer_weights = {}
            
            def collect_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.split('/')
                    if parts:
                        layer_name = parts[0]
                        if layer_name not in layer_weights:
                            layer_weights[layer_name] = []
                        layer_weights[layer_name].append({
                            'name': '/'.join(parts[1:]),
                            'shape': obj.shape
                        })
            
            weights_group.visititems(collect_weights)
            
            # 주요 레이어의 가중치 표시
            print(f"\n  총 {len(layer_weights)}개 레이어에 가중치 존재")
            
            # 처음 5개와 마지막 5개 레이어
            layer_names = list(layer_weights.keys())
            
            print("\n  [처음 5개 가중치 레이어]")
            for layer_name in layer_names[:5]:
                weights = layer_weights[layer_name]
                print(f"  - {layer_name}:")
                for w in weights:
                    print(f"      {w['name']}: {w['shape']}")
            
            if len(layer_names) > 10:
                print("\n  ... 중간 레이어 생략 ...\n")
            
            print("\n  [마지막 5개 가중치 레이어]")
            for layer_name in layer_names[-5:]:
                weights = layer_weights[layer_name]
                print(f"  - {layer_name}:")
                for w in weights:
                    print(f"      {w['name']}: {w['shape']}")
        
        # 모델 아키텍처 추정
        print("\n🏗️ 모델 아키텍처 추정:")
        
        # MobileNet 관련 레이어 확인
        mobilenet_layers = [name for name in layer_weights.keys() if 'block_' in name or 'Conv1' in name or 'expanded_conv' in name]
        if mobilenet_layers:
            print(f"  - MobileNet 기반 모델 (블록 수: {len([l for l in mobilenet_layers if 'block_' in l])})")
            
        # Dense 레이어 확인
        dense_layers = [name for name in layer_weights.keys() if 'dense' in name.lower() or 'predictions' in name]
        if dense_layers:
            print(f"  - Dense 레이어: {dense_layers}")
            
        # 최종 출력 레이어 확인
        if 'predictions' in layer_weights:
            pred_weights = layer_weights['predictions']
            for w in pred_weights:
                if 'kernel' in w['name']:
                    print(f"  - 출력 크기: {w['shape'][1]} 클래스")

def main():
    base_dir = Path("models/health_diagnosis/skin_disease/classification")
    
    models = [
        base_dir / "dog_binary/dog_binary_model.h5",
        base_dir / "cat_binary/cat_binary_model.h5",
        base_dir / "dog_multi_136/dog_multi_136_model.h5",
        base_dir / "dog_multi_456/dog_multi_456_model.h5"
    ]
    
    print("🔍 피부 질환 모델 구조 상세 분석")
    
    # 분석 결과를 저장할 딕셔너리
    all_analysis_results = {}
    
    for model_path in models:
        if model_path.exists():
            # 원래 stdout을 저장
            import io
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            # 분석 실행
            analyze_h5_model(model_path)
            
            # 결과 캡처
            analysis_output = buffer.getvalue()
            sys.stdout = old_stdout
            
            # 화면에도 출력
            print(analysis_output)
            
            # 결과 저장
            all_analysis_results[model_path.name] = analysis_output
        else:
            print(f"\n❌ 모델 파일이 없습니다: {model_path}")
            all_analysis_results[model_path.name] = "모델 파일이 없습니다"
    
    # 전체 분석 결과를 파일로 저장
    output_path = Path("skin_models_structure_analysis.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("🔍 피부 질환 모델 구조 상세 분석 결과\n")
        f.write("="*60 + "\n\n")
        
        for model_name, analysis in all_analysis_results.items():
            f.write(f"\n{'#'*60}\n")
            f.write(f"# {model_name}\n")
            f.write(f"{'#'*60}\n")
            f.write(analysis)
            f.write("\n")
    
    print("\n\n💡 분석 완료")
    print(f"📄 분석 결과가 {output_path}에 저장되었습니다.")
    print("이 정보를 바탕으로 정확한 모델 구조를 재현할 수 있습니다.")

if __name__ == "__main__":
    main()