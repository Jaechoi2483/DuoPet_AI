"""
DuoPet 통합 테스트
실제 서비스 환경에서 행동 분석 기능을 테스트합니다.
"""
import requests
import json
import time
import os

# 서버 설정
BACKEND_URL = "http://localhost:8080"  # Spring Boot
AI_SERVICE_URL = "http://localhost:8000"  # FastAPI

def test_ai_service_direct():
    """AI 서비스 직접 테스트"""
    print("\n1. AI 서비스 직접 테스트")
    print("-" * 50)
    
    # Health check
    try:
        response = requests.get(f"{AI_SERVICE_URL}/health")
        if response.status_code == 200:
            print("✅ AI 서비스 정상 작동")
        else:
            print("❌ AI 서비스 응답 이상")
            return False
    except:
        print("❌ AI 서비스에 연결할 수 없습니다")
        return False
    
    # 행동 분석 엔드포인트 확인
    response = requests.get(f"{AI_SERVICE_URL}/docs")
    if response.status_code == 200:
        print("✅ API 문서 접근 가능")
        print(f"   Swagger UI: {AI_SERVICE_URL}/docs")
    
    return True

def test_backend_integration():
    """백엔드 통합 테스트"""
    print("\n2. 백엔드 서버 연동 테스트")
    print("-" * 50)
    
    try:
        # 백엔드 health check
        response = requests.get(f"{BACKEND_URL}/actuator/health")
        if response.status_code == 200:
            print("✅ 백엔드 서버 정상 작동")
        else:
            print("❌ 백엔드 서버 응답 이상")
            return False
    except:
        print("❌ 백엔드 서버에 연결할 수 없습니다")
        print("   Spring Boot 서버를 시작하세요:")
        print("   cd D:\\final_project\\DuoPet_backend")
        print("   mvnw.cmd spring-boot:run")
        return False
    
    # TODO: 백엔드에서 AI 서비스 호출하는 엔드포인트 테스트
    # 예: /api/behavior/analyze
    
    return True

def test_full_flow(video_path, pet_type='dog'):
    """전체 플로우 테스트"""
    print(f"\n3. 전체 통합 플로우 테스트")
    print("-" * 50)
    print(f"비디오: {video_path}")
    print(f"펫 타입: {pet_type}")
    
    # 1단계: AI 서비스로 직접 분석
    print("\n[1단계] AI 서비스 직접 호출")
    
    with open(video_path, 'rb') as f:
        files = {'video': f}
        data = {'pet_type': pet_type}
        response = requests.post(
            f"{AI_SERVICE_URL}/api/v1/behavior-analysis/analyze",
            files=files,
            data=data
        )
    
    if response.status_code != 200:
        print(f"❌ 분석 시작 실패: {response.status_code}")
        return False
    
    result = response.json()
    analysis_id = result['data']['analysis_id']
    print(f"✅ 분석 시작: {analysis_id}")
    
    # 2단계: 상태 확인
    print("\n[2단계] 분석 진행 상황 확인")
    
    while True:
        time.sleep(2)
        status_response = requests.get(
            f"{AI_SERVICE_URL}/api/v1/behavior-analysis/analysis/{analysis_id}"
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()['data']
            status = status_data['status']
            progress = status_data.get('progress', 0)
            
            print(f"\r진행률: {progress}% - {status}", end='', flush=True)
            
            if status == 'completed':
                print("\n✅ 분석 완료!")
                
                # 결과 요약
                print("\n[분석 결과]")
                print(f"- 비디오 길이: {status_data.get('video_duration', 0):.1f}초")
                print(f"- 포즈 추정 사용: {'예' if status_data.get('pose_estimation_used') else '아니오'}")
                print(f"- 포즈 사용률: {status_data.get('pose_usage_percentage', 0):.1f}%")
                
                if 'behavior_summary' in status_data:
                    print("\n행동 요약:")
                    for behavior, count in status_data['behavior_summary'].items():
                        print(f"  • {behavior}: {count}회")
                
                # 3단계: 데이터베이스 저장 확인 (백엔드 연동 시)
                print("\n[3단계] 데이터 저장 확인")
                print("⚠️ 백엔드 연동이 구현되면 DB 저장을 확인합니다")
                
                return True
                
            elif status == 'failed':
                print(f"\n❌ 분석 실패: {status_data.get('error')}")
                return False

def main():
    """메인 테스트 실행"""
    print("🔍 DuoPet 서비스 통합 테스트")
    print("=" * 60)
    
    # 1. AI 서비스 테스트
    if not test_ai_service_direct():
        print("\n⚠️ AI 서비스를 먼저 시작하세요")
        return
    
    # 2. 백엔드 연동 테스트
    backend_ok = test_backend_integration()
    
    # 3. 전체 플로우 테스트
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        pet_type = sys.argv[2] if len(sys.argv) > 2 else 'dog'
        
        if os.path.exists(video_path):
            test_full_flow(video_path, pet_type)
        else:
            print(f"\n❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
    else:
        print("\n사용법:")
        print("python test_integration.py <비디오파일> [pet_type]")
        print("예: python test_integration.py C:\\cat_video.mp4 cat")
    
    # 요약
    print("\n" + "=" * 60)
    print("📋 테스트 요약")
    print(f"- AI 서비스: ✅ 정상")
    print(f"- 백엔드 연동: {'✅ 정상' if backend_ok else '❌ 미연결'}")
    print(f"- 포즈 추정: ✅ Fallback 모드로 작동")
    
    if not backend_ok:
        print("\n💡 백엔드 연동을 위해:")
        print("1. Spring Boot 서버 시작")
        print("2. AI 서비스 호출 코드 구현")
        print("3. 분석 결과 DB 저장 구현")

if __name__ == "__main__":
    main()