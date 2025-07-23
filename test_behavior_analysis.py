import requests
import json
import time
import sys

# API 서버 주소
BASE_URL = "http://localhost:8000/api/v1"

def test_video_analysis(video_path, pet_type='dog'):
    """비디오 행동 분석 테스트 (간소화 버전)"""
    print(f"\n🎥 비디오 행동 분석 시작")
    print(f"  파일: {video_path}")
    print(f"  펫 타입: {pet_type}")
    print("-" * 50)
    
    # 1. 분석 시작
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {'pet_type': pet_type}
        response = requests.post(f"{BASE_URL}/behavior-analysis/analyze", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ HTTP 에러: {response.status_code}")
        return
        
    result = response.json()
    if not result['success']:
        print(f"❌ 실패: {result.get('message', 'Unknown error')}")
        return
        
    analysis_id = result['data']['analysis_id']
    print(f"✅ 분석 시작됨! ID: {analysis_id}")
    
    # 2. 상태 확인
    print("\n📊 분석 진행 상황:")
    while True:
        time.sleep(2)
        status_response = requests.get(f"{BASE_URL}/behavior-analysis/analysis/{analysis_id}")
        
        if status_response.status_code != 200:
            print(f"❌ 상태 확인 실패")
            break
            
        status_data = status_response.json()['data']
        status = status_data['status']
        progress = status_data.get('progress', 0)
        
        if status == 'completed':
            print(f"\r✅ 분석 완료! (100%)                    ")
            print("\n🎯 분석 결과:")
            print("-" * 50)
            
            # 결과 출력
            print(f"📹 비디오 길이: {status_data['video_duration']:.1f}초")
            print(f"🤖 포즈 추정 사용: {'예' if status_data['pose_estimation_used'] else '아니오'}")
            print(f"📈 포즈 사용률: {status_data['pose_usage_percentage']:.1f}%")
            
            # 행동 요약
            if 'behavior_summary' in status_data:
                print(f"\n📊 행동 요약:")
                total_behaviors = sum(status_data['behavior_summary'].values())
                for behavior, count in status_data['behavior_summary'].items():
                    percentage = (count / total_behaviors * 100) if total_behaviors > 0 else 0
                    print(f"  • {behavior}: {count}회 ({percentage:.1f}%)")
            
            # 감지된 행동 상세
            if 'behaviors' in status_data and status_data['behaviors']:
                print(f"\n🔍 감지된 행동 상세 (총 {len(status_data['behaviors'])}개):")
                for i, behavior in enumerate(status_data['behaviors'][:5], 1):
                    print(f"  {i}. {behavior['behavior_type']} "
                          f"({behavior['start_time']:.1f}초 ~ {behavior['end_time']:.1f}초) "
                          f"신뢰도: {behavior['confidence']*100:.1f}%")
                
                if len(status_data['behaviors']) > 5:
                    print(f"  ... 외 {len(status_data['behaviors'])-5}개 더")
            
            # 이상 행동
            if status_data.get('abnormal_behaviors'):
                print(f"\n⚠️ 이상 행동 감지: {len(status_data['abnormal_behaviors'])}개")
            else:
                print(f"\n✅ 이상 행동 없음")
                
            break
            
        elif status == 'failed':
            print(f"\r❌ 분석 실패: {status_data.get('error', 'Unknown error')}")
            break
        else:
            # 진행 중
            message = status_data.get('message', '처리 중...')
            print(f"\r  {message} ({progress:.0f}%)", end='', flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        pet_type = sys.argv[2] if len(sys.argv) > 2 else 'dog'
        test_video_analysis(video_file, pet_type)
    else:
        print("사용법: python test_behavior_analysis.py <비디오파일> [pet_type]")
        print("예시: python test_behavior_analysis.py C:\\cat_video.mp4 cat")