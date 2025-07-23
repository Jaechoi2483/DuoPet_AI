#!/usr/bin/env python3
"""
실제 AI 모델 동작 테스트
"""
import requests
import json
import os
import time

# API 엔드포인트
API_URL = "http://localhost:8000/api/v1/behavior-analysis/analyze"

# 테스트 비디오 경로
TEST_VIDEO_PATH = "/mnt/d/final_project/DuoPet_AI/test_videos/sample_pet_video.mp4"

def test_behavior_analysis():
    """실제 AI 모델로 행동 분석 테스트"""
    
    # 비디오 파일 확인
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"❌ 테스트 비디오 파일이 없습니다: {TEST_VIDEO_PATH}")
        print("샘플 비디오를 생성하거나 다른 비디오 파일 경로를 지정하세요.")
        return
    
    print("="*60)
    print("실제 AI 모델 동작 테스트")
    print("="*60)
    
    # 파일 업로드
    with open(TEST_VIDEO_PATH, 'rb') as f:
        files = {'video': ('test_video.mp4', f, 'video/mp4')}
        data = {'pet_type': 'dog'}  # 강아지로 테스트
        
        print(f"✅ 비디오 파일: {TEST_VIDEO_PATH}")
        print(f"✅ 반려동물 종류: {data['pet_type']}")
        print("\n📤 API 요청 중...")
        
        start_time = time.time()
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            elapsed_time = time.time() - start_time
            
            print(f"\n⏱️ 처리 시간: {elapsed_time:.2f}초")
            print(f"📥 응답 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ 분석 성공!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                
                # 결과 요약
                if 'data' in result:
                    data = result['data']
                    print("\n📊 분석 결과 요약:")
                    print(f"- 비디오 길이: {data.get('video_duration', 0):.2f}초")
                    print(f"- 총 프레임 수: {data.get('total_frames', 0)}")
                    
                    if 'behavior_summary' in data:
                        print("\n🐕 감지된 행동:")
                        for behavior, count in data['behavior_summary'].items():
                            print(f"  - {behavior}: {count}회")
                    
                    if 'abnormal_behaviors' in data and data['abnormal_behaviors']:
                        print("\n⚠️ 비정상 행동 감지:")
                        for abnormal in data['abnormal_behaviors']:
                            print(f"  - 시간: {abnormal['time']:.2f}초, 행동: {abnormal['behavior']}, 신뢰도: {abnormal['confidence']:.2f}")
                    
                    # 실제 AI 모델 사용 여부 확인
                    if 'behavior_sequences' in data and data['behavior_sequences']:
                        seq = data['behavior_sequences'][0]
                        if 'behavior' in seq and 'all_probabilities' in seq['behavior']:
                            print("\n✅ 실제 AI 모델이 사용되고 있습니다!")
                            print("   (더미 데이터가 아닌 실제 확률 분포가 반환됨)")
                        else:
                            print("\n⚠️ 더미 모델이 사용된 것으로 보입니다.")
                    
            else:
                print(f"\n❌ 분석 실패: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.ConnectionError:
            print("\n❌ AI 서버에 연결할 수 없습니다.")
            print("서버가 실행 중인지 확인하세요.")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    test_behavior_analysis()