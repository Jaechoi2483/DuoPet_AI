import requests
import json

# API 서버 주소
BASE_URL = "http://localhost:8000/api/v1"

def test_pose_estimation(image_path):
    """포즈 추정 테스트"""
    print(f"\n=== 포즈 추정 테스트 ===")
    print(f"이미지: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/behavior-analysis/test-pose", files=files)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            data = result['data']
            print(f"✓ 성공!")
            print(f"  - 키포인트 수: {data['num_keypoints']}")
            print(f"  - 유효 키포인트: {data['valid_keypoints']}")
            print(f"  - 사용 방법: {data['method']}")
            print(f"  - 펫 감지: {data['pet_detected']}")
            if data['pet_detected']:
                print(f"  - 펫 종류: {data['pet_class']}")
        else:
            print(f"✗ 실패: {result.get('message', 'Unknown error')}")
    else:
        print(f"✗ HTTP 에러: {response.status_code}")

def test_video_analysis(video_path, pet_type='dog'):
    """비디오 행동 분석 테스트"""
    print(f"\n=== 비디오 행동 분석 테스트 ===")
    print(f"비디오: {video_path}")
    print(f"펫 타입: {pet_type}")
    
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {'pet_type': pet_type}
        response = requests.post(f"{BASE_URL}/behavior-analysis/analyze", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"응답: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if result['success']:
            # 작업 ID 받기
            if 'job_id' in result['data']:
                job_id = result['data']['job_id']
            elif 'analysis_id' in result['data']:
                job_id = result['data']['analysis_id']
            else:
                print("알 수 없는 응답 형식:")
                print(json.dumps(result['data'], indent=2, ensure_ascii=False))
                return
                
            print(f"✓ 분석 시작됨! ID: {job_id}")
            
            # 상태 확인
            import time
            while True:
                time.sleep(2)  # 2초 대기
                print(f"\n상태 확인 중... {BASE_URL}/behavior-analysis/analysis/{job_id}")
                status_response = requests.get(f"{BASE_URL}/behavior-analysis/analysis/{job_id}")
                print(f"상태 응답 코드: {status_response.status_code}")
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"상태 응답: {json.dumps(status_result, indent=2, ensure_ascii=False)}")
                    
                    if status_result['success']:
                        status = status_result['data']['status']
                        progress = status_result['data'].get('progress', 0)
                        print(f"  상태: {status} ({progress:.1f}%)")
                        
                        if status == 'completed':
                            print(f"\n✓ 분석 완료!")
                            # 응답 구조 확인
                            print("완료 응답:")
                            print(json.dumps(status_result['data'], indent=2, ensure_ascii=False))
                            
                            # result가 있는지 확인
                            if 'result' in status_result['data']:
                                analysis = status_result['data']['result']
                            else:
                                # 직접 data에서 결과 가져오기
                                analysis = status_result['data']
                            
                            # 사용 가능한 키 확인
                            if 'video_duration' in analysis:
                                print(f"  - 비디오 길이: {analysis['video_duration']:.1f}초")
                            if 'behavior_sequences' in analysis:
                                print(f"  - 감지된 행동: {len(analysis['behavior_sequences'])}개")
                            if 'pose_estimation_used' in analysis:
                                print(f"  - 포즈 추정 사용: {analysis['pose_estimation_used']}")
                            
                            # 행동 요약
                            if 'behavior_summary' in analysis:
                                print(f"\n  행동 요약:")
                                for behavior, count in list(analysis['behavior_summary'].items())[:5]:
                                    # count가 정수인 경우와 딕셔너리인 경우 모두 처리
                                    if isinstance(count, dict):
                                        print(f"    - {behavior}: {count['count']}회 ({count['percentage']:.1f}%)")
                                    else:
                                        print(f"    - {behavior}: {count}회")
                            break
                        elif status == 'failed':
                            print(f"\n✗ 분석 실패: {status_result['data'].get('error', 'Unknown error')}")
                            break
                else:
                    print(f"상태 확인 실패: {status_response.text}")
                    break
        else:
            print(f"✗ 실패: {result.get('message', 'Unknown error')}")
    else:
        print(f"✗ HTTP 에러: {response.status_code}")

if __name__ == "__main__":
    import sys
    import os
    
    print("DuoPet AI 행동 분석 API 테스트")
    print("================================")
    
    # 테스트할 파일 확인
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if test_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            test_pose_estimation(test_file)
        elif test_file.lower().endswith(('.mp4', '.avi', '.mov')):
            pet_type = sys.argv[2] if len(sys.argv) > 2 else 'dog'
            test_video_analysis(test_file, pet_type)
        else:
            print("지원하지 않는 파일 형식입니다.")
    else:
        print("\n사용법:")
        print("  python test_api_simple.py <이미지_또는_비디오_파일> [pet_type]")
        print("\n예시:")
        print("  python test_api_simple.py D:\\pet_image.jpg")
        print("  python test_api_simple.py D:\\pet_video.mp4 cat")