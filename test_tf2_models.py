"""
TensorFlow 2.x 변환 모델 테스트
각 모델이 제대로 로드되고 예측이 가능한지 확인
"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
from typing import Dict, Any

# TensorFlow 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelTester:
    """모델 테스터"""
    
    def __init__(self):
        self.models_dir = Path("models/health_diagnosis")
        self.test_results = []
        
        # 테스트할 모델 목록 (각 유형별 하나씩)
        self.models_to_test = {
            "eye_disease": {
                "paths": [
                    "eye_disease/eye_disease_tf2_complete.h5",
                    "eye_disease/eye_disease_fixed.h5",
                    "eye_disease/best_grouped_model_fixed.h5"
                ],
                "input_shape": (1, 224, 224, 3),
                "expected_output_shape": (1, 5),
                "preprocessing": "efficientnet"
            },
            "bcs": {
                "paths": [
                    "bcs/bcs_tf2_unified.h5",
                    "bcs/bcs_efficientnet_v1.h5"
                ],
                "input_shape": (1, 224, 224, 3),
                "expected_output_shape": (1, 3),
                "preprocessing": "standard"
            },
            "skin_cat_binary": {
                "paths": [
                    "skin_disease/classification/cat_binary/cat_binary_tf2_unified.h5",
                    "skin_disease/classification/cat_binary/cat_binary_model_tf2_perfect.h5",
                    "skin_disease/classification/cat_binary/cat_binary_model.h5"
                ],
                "input_shape": (1, 224, 224, 3),
                "expected_output_shape": (1, 1),
                "preprocessing": "mobilenet"
            },
            "skin_dog_binary": {
                "paths": [
                    "skin_disease/classification/dog_binary/dog_binary_tf2_unified.h5",
                    "skin_disease/classification/dog_binary/dog_binary_model_tf2_perfect.h5",
                    "skin_disease/classification/dog_binary/dog_binary_model.h5"
                ],
                "input_shape": (1, 224, 224, 3),
                "expected_output_shape": (1, 1),
                "preprocessing": "mobilenet"
            }
        }
    
    def preprocess_input(self, input_data: np.ndarray, preprocessing: str) -> np.ndarray:
        """입력 전처리"""
        if preprocessing == "efficientnet":
            return tf.keras.applications.efficientnet.preprocess_input(input_data)
        elif preprocessing == "mobilenet":
            return tf.keras.applications.mobilenet_v2.preprocess_input(input_data)
        else:  # standard
            return input_data / 255.0
    
    def test_model(self, model_name: str, config: Dict) -> Dict[str, Any]:
        """개별 모델 테스트"""
        print(f"\n{'='*60}")
        print(f"🧪 {model_name} 모델 테스트")
        print(f"{'='*60}")
        
        result = {
            "model_name": model_name,
            "status": "not_found",
            "working_path": None,
            "load_time": None,
            "inference_time": None,
            "output_shape": None,
            "errors": []
        }
        
        # 사용 가능한 모델 찾기
        model_path = None
        for path in config["paths"]:
            full_path = self.models_dir / path
            if full_path.exists():
                model_path = full_path
                print(f"✅ 모델 파일 발견: {path}")
                break
        
        if not model_path:
            print(f"❌ 모델 파일을 찾을 수 없습니다")
            result["errors"].append("No model file found")
            return result
        
        try:
            # 1. 모델 로드 테스트
            print("\n1️⃣ 모델 로드 테스트...")
            start_time = time.time()
            
            # Custom objects
            custom_objects = {
                'swish': tf.nn.swish,
                'Swish': tf.nn.swish,
            }
            
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False
            )
            
            load_time = time.time() - start_time
            result["load_time"] = load_time
            print(f"   ✓ 로드 시간: {load_time:.2f}초")
            
            # 2. 모델 구조 확인
            print("\n2️⃣ 모델 구조 확인...")
            print(f"   - 입력 shape: {model.input_shape}")
            print(f"   - 출력 shape: {model.output_shape}")
            print(f"   - 총 레이어 수: {len(model.layers)}")
            print(f"   - 파라미터 수: {model.count_params():,}")
            
            # 3. 추론 테스트
            print("\n3️⃣ 추론 테스트...")
            
            # 테스트 입력 생성
            test_input = np.random.randint(0, 255, config["input_shape"], dtype=np.uint8)
            test_input = test_input.astype(np.float32)
            
            # 전처리
            processed_input = self.preprocess_input(test_input, config["preprocessing"])
            
            # 추론
            start_time = time.time()
            output = model.predict(processed_input, verbose=0)
            inference_time = time.time() - start_time
            
            result["inference_time"] = inference_time
            result["output_shape"] = output.shape
            
            print(f"   ✓ 추론 시간: {inference_time*1000:.2f}ms")
            print(f"   ✓ 출력 shape: {output.shape}")
            
            # 출력 검증
            if output.shape == config["expected_output_shape"]:
                print(f"   ✓ 출력 shape 정상")
            else:
                print(f"   ⚠️ 예상과 다른 출력 shape")
                result["errors"].append(f"Unexpected output shape: {output.shape}")
            
            # 4. 출력 값 확인
            print("\n4️⃣ 출력 값 확인...")
            if len(output.shape) == 2 and output.shape[1] == 1:
                # Binary classification
                print(f"   - 확률: {output[0][0]:.4f}")
                print(f"   - 예측: {'Positive' if output[0][0] > 0.5 else 'Negative'}")
            else:
                # Multi-class
                print(f"   - 확률 분포: {output[0]}")
                print(f"   - 예측 클래스: {np.argmax(output[0])}")
                print(f"   - 최대 확률: {np.max(output[0]):.4f}")
            
            # 5. 메모리 정리
            del model
            tf.keras.backend.clear_session()
            
            result["status"] = "success"
            result["working_path"] = str(model_path.relative_to(self.models_dir))
            
            print(f"\n✅ {model_name} 테스트 성공!")
            
        except Exception as e:
            print(f"\n❌ 테스트 실패: {e}")
            result["status"] = "failed"
            result["errors"].append(str(e))
        
        return result
    
    def run_all_tests(self):
        """모든 모델 테스트 실행"""
        print("🚀 TensorFlow 2.x 모델 테스트 시작")
        print(f"TensorFlow 버전: {tf.__version__}")
        print(f"Eager execution: {tf.executing_eagerly()}")
        
        # 각 모델 테스트
        for model_name, config in self.models_to_test.items():
            result = self.test_model(model_name, config)
            self.test_results.append(result)
            
            # 모델 간 충돌 방지를 위한 대기
            time.sleep(1)
        
        # 결과 요약
        self.print_summary()
    
    def print_summary(self):
        """테스트 결과 요약"""
        print(f"\n{'='*60}")
        print("📊 테스트 결과 요약")
        print(f"{'='*60}")
        
        success_count = sum(1 for r in self.test_results if r["status"] == "success")
        failed_count = sum(1 for r in self.test_results if r["status"] == "failed")
        not_found_count = sum(1 for r in self.test_results if r["status"] == "not_found")
        
        print(f"\n총 테스트: {len(self.test_results)}개")
        print(f"✅ 성공: {success_count}개")
        print(f"❌ 실패: {failed_count}개")
        print(f"⚠️ 파일 없음: {not_found_count}개")
        
        print("\n상세 결과:")
        print("-" * 60)
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"\n{status_icon} {result['model_name']}:")
            print(f"   - 상태: {result['status']}")
            
            if result["working_path"]:
                print(f"   - 작동 모델: {result['working_path']}")
            
            if result["load_time"]:
                print(f"   - 로드 시간: {result['load_time']:.2f}초")
            
            if result["inference_time"]:
                print(f"   - 추론 시간: {result['inference_time']*1000:.2f}ms")
            
            if result["errors"]:
                print(f"   - 오류: {', '.join(result['errors'])}")
        
        # 권장사항
        print(f"\n{'='*60}")
        print("💡 권장사항:")
        print(f"{'='*60}")
        
        for result in self.test_results:
            if result["status"] == "success":
                print(f"\n✅ {result['model_name']}: {result['working_path']} 사용 권장")
            elif result["status"] == "not_found":
                print(f"\n⚠️ {result['model_name']}: 모델 변환 필요")
                print(f"   - advanced_eye_model_converter.py 실행 권장")

if __name__ == "__main__":
    tester = ModelTester()
    tester.run_all_tests()