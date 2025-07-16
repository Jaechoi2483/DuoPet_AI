# test_openai.py (최종 진단 및 해결 코드)
import os
import httpx
import functools
from openai import OpenAI
from dotenv import load_dotenv

print("OpenAI 라이브러리 초기화 최종 해결 테스트를 시작합니다...")
print("문제가 되는 'proxies' 인수를 강제로 무시하도록 라이브러리 동작을 수정합니다.")

# --- 라이브러리 동작 수정 (Monkey Patching) ---
# httpx.Client의 원래 초기화 함수를 저장합니다.
original_init = httpx.Client.__init__


# 'proxies' 인수를 가로채서 제거하는 새로운 함수를 정의합니다.
@functools.wraps(original_init)
def patched_init(self, *args, **kwargs):
    if 'proxies' in kwargs:
        print("▶ 'proxies' 인수를 감지하여 강제로 제거했습니다.")
        del kwargs['proxies']
    # 수정된 인수로 원래 초기화 함수를 호출합니다.
    original_init(self, *args, **kwargs)


# httpx.Client의 초기화 함수를 우리가 만든 새 함수로 교체합니다.
httpx.Client.__init__ = patched_init
# ---------------------------------------------

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("🚨 .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
else:
    try:
        # 이제 수정된 라이브러리 동작 하에 클라이언트를 초기화합니다.
        client = OpenAI(api_key=api_key)

        print("\n✅ 성공: OpenAI 클라이언트가 성공적으로 초기화되었습니다!")
        print("환경 문제를 코드로 우회하는 데 성공했습니다.")
        print("이 해결책을 predict.py에 최종 적용하여 전달해 드리겠습니다.")

    except Exception as e:
        print(f"\n🚨 실패: 최종 해결책으로도 오류가 발생했습니다.")
        print(f"오류 유형: {type(e).__name__}")
        print(f"오류 메시지: {e}")
        print("\n이것은 해결이 매우 어려운 심각한 시스템 환경 문제입니다.")