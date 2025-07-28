"""
피부 질환 모델 분석 스크립트
기존 모델들의 구조와 상태를 확인합니다.
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

def analyze_skin_models():
    """피부 질환 모델들을 분석"""
    
    base_dir = Path("models/health_diagnosis/skin_disease")
    print("🔍 피부 질환 모델 분석 시작...\n")
    
    # 분류 모델 경로들
    classification_models = {
        "dog_binary": base_dir / "classification/dog_binary/dog_binary_model.h5",
        "cat_binary": base_dir / "classification/cat_binary/cat_binary_model.h5",
        "dog_multi_136": base_dir / "classification/dog_multi_136/dog_multi_136_model.h5",
        "dog_multi_456": base_dir / "classification/dog_multi_456/dog_multi_456_model.h5"
    }
    
    # 세그멘테이션 모델 경로들
    segmentation_models = {
        "dog_A1": base_dir / "segmentation/dog_A1",
        "dog_A2": base_dir / "segmentation/dog_A2",
        "dog_A3": base_dir / "segmentation/dog_A3",
        "dog_A4": base_dir / "segmentation/dog_A4",
        "dog_A5": base_dir / "segmentation/dog_A5",
        "dog_A6": base_dir / "segmentation/dog_A6",
        "cat_A2": base_dir / "segmentation/cat_A2"
    }
    
    print("=" * 60)
    print("📊 분류 모델 분석")
    print("=" * 60)
    
    analysis_results = {"classification": {}, "segmentation": {}}
    
    for name, model_path in classification_models.items():
        print(f"\n🔹 {name} 모델 분석")
        print(f"   경로: {model_path}")
        
        if not model_path.exists():
            print(f"   ❌ 파일이 존재하지 않습니다!")
            continue
            
        try:
            # 파일 크기 확인
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   📁 파일 크기: {file_size:.2f} MB")
            
            # 모델 로드 시도
            print(f"   🔄 모델 로드 중...")
            model = tf.keras.models.load_model(str(model_path), compile=False)
            print(f"   ✅ 모델 로드 성공!")
            
            # 모델 구조 분석
            print(f"\n   📐 모델 구조:")
            print(f"   - 입력 shape: {model.input_shape}")
            print(f"   - 출력 shape: {model.output_shape}")
            print(f"   - 총 레이어 수: {len(model.layers)}")
            print(f"   - 총 파라미터: {model.count_params():,}")
            
            # 레이어 타입 분석
            layer_types = {}
            for layer in model.layers:
                layer_type = type(layer).__name__
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            
            print(f"\n   🔧 레이어 구성:")
            for layer_type, count in layer_types.items():
                print(f"   - {layer_type}: {count}개")
            
            # 테스트 예측
            print(f"\n   🧪 테스트 예측:")
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            try:
                # TF 1.x 스타일 예측
                predictions = model.predict(test_input, verbose=0)
                print(f"   ✅ 예측 성공!")
                print(f"   - 출력 shape: {predictions.shape}")
                print(f"   - 출력 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
                print(f"   - 출력 합계: {predictions.sum():.4f}")
                
                # 활성화 함수 추정
                if abs(predictions.sum() - 1.0) < 0.01:
                    print(f"   - 추정 활성화: softmax (합계 ≈ 1)")
                elif predictions.max() <= 1.0 and predictions.min() >= 0.0:
                    print(f"   - 추정 활성화: sigmoid (0-1 범위)")
                else:
                    print(f"   - 추정 활성화: 기타")
                    
            except Exception as e:
                print(f"   ❌ 예측 실패: {e}")
            
            # 결과 저장
            analysis_results["classification"][name] = {
                "exists": True,
                "file_size_mb": file_size,
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "total_params": model.count_params(),
                "loadable": True,
                "predictable": predictions is not None if 'predictions' in locals() else False
            }
            
        except Exception as e:
            print(f"   ❌ 모델 분석 실패: {e}")
            analysis_results["classification"][name] = {
                "exists": True,
                "loadable": False,
                "error": str(e)
            }
    
    print("\n" + "=" * 60)
    print("📊 세그멘테이션 모델 분석")
    print("=" * 60)
    
    for name, model_path in segmentation_models.items():
        print(f"\n🔹 {name} 모델 분석")
        print(f"   경로: {model_path}")
        
        if not model_path.exists():
            print(f"   ❌ 디렉토리가 존재하지 않습니다!")
            analysis_results["segmentation"][name] = {"exists": False}
            continue
            
        # 체크포인트 파일 확인 - 여러 패턴 검색
        data_files = list(model_path.glob("*.data-*"))
        index_files = list(model_path.glob("*.index"))
        checkpoint_files = list(model_path.glob("checkpoint"))
        
        all_checkpoint_files = data_files + index_files + checkpoint_files
        
        if all_checkpoint_files:
            print(f"   ✅ 체크포인트 파일 발견: {len(all_checkpoint_files)}개")
            # 파일 타입별로 분류해서 표시
            if data_files:
                print(f"      📊 데이터 파일: {len(data_files)}개")
                for f in data_files[:2]:
                    print(f"         - {f.name}")
            if index_files:
                print(f"      📑 인덱스 파일: {len(index_files)}개")
                for f in index_files[:2]:
                    print(f"         - {f.name}")
            if checkpoint_files:
                print(f"      📌 체크포인트 파일: {len(checkpoint_files)}개")
                for f in checkpoint_files:
                    print(f"         - {f.name}")
        else:
            print(f"   ❌ 체크포인트 파일이 없습니다!")
            
        analysis_results["segmentation"][name] = {
            "exists": True,
            "checkpoint_count": len(all_checkpoint_files),
            "data_files": len(data_files),
            "index_files": len(index_files),
            "checkpoint_files": len(checkpoint_files)
        }
    
    # 분석 결과 저장
    result_path = Path("skin_models_analysis.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n{'='*60}")
    print("📊 분석 요약")
    print("="*60)
    
    # 분류 모델 요약
    total_classification = len(classification_models)
    loadable_classification = sum(1 for r in analysis_results["classification"].values() 
                                if r.get("loadable", False))
    
    print(f"\n분류 모델:")
    print(f"  - 전체: {total_classification}개")
    print(f"  - 로드 가능: {loadable_classification}개")
    
    # 세그멘테이션 모델 요약
    total_segmentation = len(segmentation_models)
    existing_segmentation = sum(1 for r in analysis_results["segmentation"].values() 
                              if r.get("exists", False))
    
    print(f"\n세그멘테이션 모델:")
    print(f"  - 전체: {total_segmentation}개")
    print(f"  - 존재함: {existing_segmentation}개")
    
    print(f"\n💾 분석 결과가 {result_path}에 저장되었습니다.")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_skin_models()
    
    # 변환 가능 여부 판단
    print("\n\n🎯 변환 가능성 평가:")
    
    can_convert = False
    for name, info in results["classification"].items():
        if info.get("loadable", False):
            print(f"  ✅ {name}: TF 2.x로 변환 가능")
            can_convert = True
        else:
            print(f"  ❌ {name}: 변환 불가능")
    
    if can_convert:
        print("\n✨ 일부 모델은 TF 2.x로 변환 가능합니다!")
        print("💡 fix_skin_model_tf2_correct.py를 실행하여 변환할 수 있습니다.")
    else:
        print("\n⚠️ 변환 가능한 모델이 없습니다.")