"""
전체 AI 모델 통합 테스트
TF2 변환된 모든 건강 진단 모델 테스트
"""
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time

# TensorFlow 2.x 설정
tf.config.run_functions_eagerly(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow 버전: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

class IntegratedModelTester:
    """통합 모델 테스터"""
    
    def __init__(self):
        self.results = {}
        self.models_base = Path("models/health_diagnosis")
        
    def test_eye_disease_model(self):
        """안구질환 모델 테스트"""
        print("\n" + "="*60)
        print("🔍 안구질환 모델 테스트")
        
        try:
            # 모델 경로
            model_path = self.models_base / "eye_disease" / "eye_disease_fixed.h5"
            
            if not model_path.exists():
                print(f"  ❌ 모델 파일 없음: {model_path}")
                self.results['eye_disease'] = {'status': 'failed', 'error': 'File not found'}
                return
            
            # 모델 로드
            print(f"  📥 모델 로드 중: {model_path}")
            start_time = time.time()
            
            custom_objects = {'swish': tf.nn.swish}
            model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
            
            load_time = time.time() - start_time
            print(f"  ✓ 로드 완료 ({load_time:.2f}초)")
            
            # 테스트 입력
            test_input = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32)
            
            # 예측
            start_time = time.time()
            predictions = model.predict(test_input, verbose=0)
            pred_time = time.time() - start_time
            
            # 결과
            classes = ['정상', '백내장', '녹내장', '망막질환', '결막염']
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            print(f"  📊 예측 결과:")
            print(f"     - 클래스: {classes[class_idx]}")
            print(f"     - 신뢰도: {confidence:.2%}")
            print(f"     - 예측 시간: {pred_time:.3f}초")
            
            self.results['eye_disease'] = {
                'status': 'success',
                'model_path': str(model_path),
                'load_time': load_time,
                'prediction_time': pred_time,
                'test_result': classes[class_idx]
            }
            
        except Exception as e:
            print(f"  ❌ 테스트 실패: {e}")
            self.results['eye_disease'] = {'status': 'failed', 'error': str(e)}
    
    def test_skin_disease_models(self):
        """피부질환 모델 테스트"""
        print("\n" + "="*60)
        print("🔍 피부질환 모델 테스트")
        
        # 테스트할 모델 목록
        skin_models = {
            'cat_binary': 'cat_binary/cat_binary_model_tf2_perfect.h5',
            'dog_binary': 'dog_binary/dog_binary_model_tf2_perfect.h5',
            'dog_multi_136': 'dog_multi_136/dog_multi_136_model_tf2_perfect.h5',
            'dog_multi_456': 'dog_multi_456/dog_multi_456_model_tf2_perfect.h5'
        }
        
        self.results['skin_disease'] = {}
        
        for model_name, model_file in skin_models.items():
            print(f"\n  📋 {model_name} 테스트:")
            
            try:
                model_path = self.models_base / "skin_disease" / "classification" / model_file
                
                if not model_path.exists():
                    # Fallback to original model
                    alt_path = model_path.parent / model_file.replace('_tf2_perfect', '')
                    if alt_path.exists():
                        model_path = alt_path
                    else:
                        print(f"    ❌ 모델 파일 없음")
                        self.results['skin_disease'][model_name] = {'status': 'failed', 'error': 'File not found'}
                        continue
                
                # 모델 로드
                start_time = time.time()
                model = tf.keras.models.load_model(str(model_path), compile=False)
                
                # 컴파일
                if 'binary' in model_name:
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                else:
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                load_time = time.time() - start_time
                
                # 테스트 입력
                test_input = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8).astype(np.float32)
                
                # 예측
                start_time = time.time()
                predictions = model.predict(test_input, verbose=0)
                pred_time = time.time() - start_time
                
                # 결과 출력
                if 'binary' in model_name:
                    has_disease = float(predictions[0][0]) > 0.5
                    print(f"    ✓ 질환 여부: {'있음' if has_disease else '없음'} ({float(predictions[0][0]):.2%})")
                else:
                    if '136' in model_name:
                        classes = ['구진플라크', '무증상', '농포여드름']
                    else:
                        classes = ['과다색소침착', '결절종괴', '미란궤양']
                    class_idx = np.argmax(predictions[0])
                    print(f"    ✓ 예측: {classes[class_idx]} ({predictions[0][class_idx]:.2%})")
                
                print(f"    ✓ 로드 시간: {load_time:.2f}초, 예측 시간: {pred_time:.3f}초")
                
                self.results['skin_disease'][model_name] = {
                    'status': 'success',
                    'load_time': load_time,
                    'prediction_time': pred_time
                }
                
            except Exception as e:
                print(f"    ❌ 테스트 실패: {e}")
                self.results['skin_disease'][model_name] = {'status': 'failed', 'error': str(e)}
    
    def test_bcs_model(self):
        """BCS 모델 테스트"""
        print("\n" + "="*60)
        print("🔍 BCS 모델 테스트")
        
        try:
            # 래퍼 사용 테스트
            print("\n  📋 래퍼 클래스 테스트:")
            
            # 래퍼 import - 올바른 경로 설정
            import sys
            bcs_path = str(self.models_base / "bcs")
            if bcs_path not in sys.path:
                sys.path.insert(0, bcs_path)
            
            # 래퍼 클래스 import
            from bcs_ensemble_wrapper import BCSEnsembleModel
            
            # 모델 로드
            start_time = time.time()
            bcs_model = BCSEnsembleModel()
            load_time = time.time() - start_time
            print(f"    ✓ 모델 로드 완료 ({load_time:.2f}초)")
            
            # 테스트 입력
            test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # 예측 (augmentation 없이)
            start_time = time.time()
            result = bcs_model.predict(test_input, augment=False)
            pred_time = time.time() - start_time
            
            print(f"    ✓ 예측 결과: {result['class']} ({result['confidence']:.2%})")
            print(f"    ✓ 예측 시간: {pred_time:.3f}초")
            
            # 직접 모델 테스트
            print("\n  📋 직접 모델 테스트:")
            model_path = self.models_base / "bcs" / "bcs_tf2_ensemble.h5"
            
            if model_path.exists():
                custom_objects = {
                    'swish': tf.nn.swish,
                    'Swish': tf.keras.layers.Activation(tf.nn.swish)
                }
                
                model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
                
                # 13개 입력 준비
                inputs_13 = [test_input.reshape(1, 224, 224, 3).astype(np.float32) for _ in range(13)]
                
                # 예측
                predictions = model.predict(inputs_13, verbose=0)
                
                classes = ['마른 체형', '정상 체형', '비만 체형']
                class_idx = np.argmax(predictions[0])
                
                print(f"    ✓ 직접 예측: {classes[class_idx]} ({predictions[0][class_idx]:.2%})")
            
            self.results['bcs'] = {
                'status': 'success',
                'wrapper_load_time': load_time,
                'prediction_time': pred_time,
                'test_result': result['class']
            }
            
        except Exception as e:
            print(f"  ❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            self.results['bcs'] = {'status': 'failed', 'error': str(e)}
    
    def generate_report(self):
        """테스트 리포트 생성"""
        print("\n" + "="*80)
        print("📊 통합 테스트 결과 요약")
        print("="*80)
        
        # 성공/실패 카운트
        total_tests = 0
        successful_tests = 0
        
        for category, results in self.results.items():
            if isinstance(results, dict):
                if 'status' in results:
                    total_tests += 1
                    if results['status'] == 'success':
                        successful_tests += 1
                else:
                    # 서브 모델들 (피부질환)
                    for sub_model, sub_results in results.items():
                        total_tests += 1
                        if sub_results.get('status') == 'success':
                            successful_tests += 1
        
        print(f"\n✅ 성공: {successful_tests}/{total_tests} 테스트")
        print(f"❌ 실패: {total_tests - successful_tests}/{total_tests} 테스트")
        
        # 상세 결과
        print("\n📋 상세 결과:")
        
        # 안구질환
        eye_result = self.results.get('eye_disease', {})
        print(f"\n  👁️ 안구질환 모델: {eye_result.get('status', 'not tested')}")
        if eye_result.get('status') == 'success':
            print(f"     - 로드 시간: {eye_result.get('load_time', 0):.2f}초")
            print(f"     - 예측 시간: {eye_result.get('prediction_time', 0):.3f}초")
        
        # 피부질환
        skin_results = self.results.get('skin_disease', {})
        print(f"\n  🐾 피부질환 모델:")
        for model_name, result in skin_results.items():
            print(f"     - {model_name}: {result.get('status', 'not tested')}")
        
        # BCS
        bcs_result = self.results.get('bcs', {})
        print(f"\n  📏 BCS 모델: {bcs_result.get('status', 'not tested')}")
        if bcs_result.get('status') == 'success':
            print(f"     - 로드 시간: {bcs_result.get('wrapper_load_time', 0):.2f}초")
            print(f"     - 예측 시간: {bcs_result.get('prediction_time', 0):.3f}초")
        
        # 결과 저장
        report_path = Path("test_results_integrated.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 상세 리포트 저장: {report_path}")
        
        # 권장사항
        print("\n💡 권장사항:")
        if successful_tests == total_tests:
            print("  ✅ 모든 모델이 정상 작동합니다!")
            print("  ✅ 서비스 재시작 후 실제 API 테스트를 진행하세요.")
        else:
            print("  ⚠️ 일부 모델에 문제가 있습니다.")
            print("  ⚠️ 실패한 모델을 확인하고 수정하세요.")

def main():
    """메인 함수"""
    print("🚀 DuoPet AI 모델 통합 테스트")
    print("=" * 80)
    
    tester = IntegratedModelTester()
    
    # 각 모델 테스트
    tester.test_eye_disease_model()
    tester.test_skin_disease_models()
    tester.test_bcs_model()
    
    # 리포트 생성
    tester.generate_report()
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()