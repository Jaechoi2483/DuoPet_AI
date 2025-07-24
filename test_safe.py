import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_server_health():
    """서버 상태 확인"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 서버가 정상 작동 중입니다.")
            return True
        else:
            print(f"❌ 서버 응답 이상: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 서버에 연결할 수 없습니다: {str(e)}")
        return False

def test_pose_endpoint():
    """포즈 엔드포인트 테스트"""
    print("\n📸 포즈 추정 테스트")
    
    # 간단한 테스트 이미지 생성
    import numpy as np
    import cv2
    import tempfile
    import os
    
    # 테스트 이미지 생성
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (200, 150), (440, 350), (128, 128, 128), -1)
    
    temp_path = os.path.join(tempfile.gettempdir(), "test_pet.jpg")
    cv2.imwrite(temp_path, test_image)
    
    try:
        with open(temp_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{BASE_URL}/behavior-analysis/test-pose", 
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                data = result['data']
                print(f"✅ 포즈 추정 성공!")
                print(f"  - 키포인트: {data.get('num_keypoints', 0)}개")
                print(f"  - 방법: {data.get('method', 'unknown')}")
                return True
        
        print(f"❌ 포즈 추정 실패: {response.status_code}")
        return False
        
    except Exception as e:
        print(f"❌ 포즈 테스트 중 오류: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_video_analysis_safe(video_path, pet_type='dog'):
    """안전한 비디오 분석 테스트"""
    print(f"\n🎥 비디오 행동 분석 테스트")
    print(f"  파일: {video_path}")
    print(f"  펫 타입: {pet_type}")
    
    try:
        # 1. 분석 시작
        with open(video_path, 'rb') as f:
            files = {'video': f}
            data = {'pet_type': pet_type}
            response = requests.post(
                f"{BASE_URL}/behavior-analysis/analyze", 
                files=files, 
                data=data,
                timeout=60  # 60초 타임아웃
            )
        
        if response.status_code != 200:
            print(f"❌ 분석 시작 실패: {response.status_code}")
            return False
            
        result = response.json()
        if not result['success']:
            print(f"❌ 분석 시작 실패: {result.get('message')}")
            return False
            
        analysis_id = result['data']['analysis_id']
        print(f"✅ 분석 시작됨! ID: {analysis_id}")
        
        # 2. 상태 확인 (타임아웃 포함)
        max_wait_time = 300  # 최대 5분 대기
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                print("\n❌ 분석 시간 초과 (5분)")
                return False
                
            time.sleep(3)  # 3초 대기
            
            try:
                status_response = requests.get(
                    f"{BASE_URL}/behavior-analysis/analysis/{analysis_id}",
                    timeout=10
                )
                
                if status_response.status_code != 200:
                    print(f"\n❌ 상태 확인 실패: {status_response.status_code}")
                    return False
                    
                status_data = status_response.json()['data']
                status = status_data['status']
                progress = status_data.get('progress', 0)
                
                print(f"\r진행률: {progress:.0f}% - {status_data.get('message', '처리 중...')}", end='', flush=True)
                
                if status == 'completed':
                    print(f"\n✅ 분석 완료!")
                    
                    # 결과 출력
                    print(f"\n📊 분석 결과:")
                    print(f"  - 비디오 길이: {status_data.get('video_duration', 0):.1f}초")
                    print(f"  - 포즈 추정: {'사용' if status_data.get('pose_estimation_used') else '미사용'}")
                    
                    if 'behavior_summary' in status_data:
                        print(f"\n  행동 요약:")
                        for behavior, count in status_data['behavior_summary'].items():
                            print(f"    • {behavior}: {count}회")
                    
                    return True
                    
                elif status == 'failed':
                    print(f"\n❌ 분석 실패: {status_data.get('error')}")
                    return False
                    
            except requests.exceptions.Timeout:
                print("\n⚠️ 상태 확인 타임아웃, 재시도...")
                continue
            except Exception as e:
                print(f"\n❌ 상태 확인 중 오류: {str(e)}")
                return False
                
    except Exception as e:
        print(f"\n❌ 비디오 분석 중 오류: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 DuoPet AI 안전 테스트")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    if not test_server_health():
        print("\n서버를 먼저 시작해주세요:")
        print("python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        sys.exit(1)
    
    # 2. 포즈 엔드포인트 테스트
    test_pose_endpoint()
    
    # 3. 비디오 분석 테스트
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        pet_type = sys.argv[2] if len(sys.argv) > 2 else 'dog'
        test_video_analysis_safe(video_file, pet_type)
    else:
        print("\n비디오 테스트를 하려면:")
        print("python test_safe.py <비디오파일> [pet_type]")