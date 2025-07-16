# --------------------------------------------------------------------------
# 파일명: services/chatbot/predict.py
# 설명: 사이트 기능 목록에 공지사항 및 자유게시판을 추가하여 안내 기능 강화
# --------------------------------------------------------------------------
import os
import json
import httpx
import functools
import time
import chromadb
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
from keybert import KeyBERT
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Dict, Any

# --- 환경 변수 로드 및 라이브러리 동작 수정 ---
load_dotenv()

original_init = httpx.Client.__init__


@functools.wraps(original_init)
def patched_init(self, *args, **kwargs):
    if 'proxies' in kwargs:
        del kwargs['proxies']
    original_init(self, *args, **kwargs)


httpx.Client.__init__ = patched_init

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY가 .env 파일에 설정되지 않았거나 비어 있습니다.")

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"🚨 OpenAI 클라이언트 초기화 실패: {e}")


# --- RAG 챗봇 클래스 정의 ---
class RAGChatbot:
    # ❗❗❗ __init__ 메서드에 SITE_FUNCTIONS를 직접 정의하도록 수정 ❗❗❗
    def __init__(self, site_url: str, max_crawl_pages: int = 10):
        print("🤖 RAG 챗봇 초기화를 시작합니다...")
        self.site_url = site_url
        # ❗ 사이트 기능 목록을 클래스 내부에 정의합니다.
        self.site_functions = [
            {"name": "notice_board", "description": "공지사항 확인하기", "url": "/notice"},
            {"name": "free_board", "description": "자유게시판 가기", "url": "/board"},
            {"name": "health_check", "description": "반려동물 건강 진단하기", "url": "/health-check"},
            {"name": "behavior_analysis", "description": "이상행동 분석 서비스 보기", "url": "/behavior-analysis"},
            {"name": "video_recommend", "description": "추천 영상 보러가기", "url": "/recommendations"}
        ]
        self.base_url = f"{urlparse(self.site_url).scheme}://{urlparse(self.site_url).netloc}"
        self.max_crawl_pages = max_crawl_pages

        print("KeyBERT 모델을 로딩 중입니다...")
        self.kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        print("모델 로딩 완료.")

        self.knowledge_base = self._create_kb_from_site()
        if not self.knowledge_base:
            raise RuntimeError("지식 베이스 생성에 실패했습니다. URL과 사이트 내용을 확인해주세요.")

        self.db_collection = self._setup_vector_db()

    def _get_page_content(self, url: str) -> str:
        """Selenium을 사용해 단일 페이지의 HTML 콘텐츠를 가져옵니다."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            driver.get(url)
            time.sleep(3)
            return driver.page_source
        except Exception as e:
            print(f"🚨 '{url}' 페이지 크롤링 중 오류 발생: {e}")
            return ""
        finally:
            if driver:
                driver.quit()

    def _create_kb_from_site(self) -> List[Dict[str, Any]]:
        """사이트를 재귀적으로 크롤링하여 지식 베이스를 구축하고 상세 로그를 출력합니다."""
        print(f"--- 🌐 사이트 전체 콘텐츠 추출 시작 (최대 {self.max_crawl_pages} 페이지) ---")

        urls_to_visit = {self.site_url}
        visited_urls = set()
        knowledge_base = []

        while urls_to_visit and len(visited_urls) < self.max_crawl_pages:
            current_url = urls_to_visit.pop()
            if current_url in visited_urls:
                continue

            print(f"\n[크롤링 시작] -> {current_url}")
            visited_urls.add(current_url)
            html_content = self._get_page_content(current_url)
            if not html_content:
                print("  [결과] 페이지 콘텐츠를 가져오지 못했습니다.")
                continue

            soup = BeautifulSoup(html_content, 'html.parser')
            page_title = soup.title.string.strip() if soup.title else '제목 없음'
            print(f"  [페이지 제목] {page_title}")

            content_area = soup.find('main') or soup.find('article') or soup.find('body')
            if not content_area:
                print("  [결과] 주요 콘텐츠 영역을 찾지 못했습니다.")
                continue

            chunks_from_page = []
            for element in content_area.find_all(['h1', 'h2', 'h3', 'p', 'div', 'li', 'span', 'a'], recursive=True):
                if isinstance(element, NavigableString): continue
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 20 and '\n' not in text:
                    chunks_from_page.append(text)

            unique_chunks = list(dict.fromkeys(chunks_from_page))

            for i, chunk in enumerate(unique_chunks):
                knowledge_base.append({
                    "id": f"{urlparse(current_url).path.replace('/', '_')}_{i}",
                    "content": chunk,
                    "metadata": {"source": current_url, "title": page_title}
                })

            print(f"  [추출된 정보] {len(unique_chunks)}개의 텍스트 조각")

            found_links = set()
            for link in content_area.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.base_url, href)
                if full_url.startswith(self.base_url) and full_url not in visited_urls:
                    found_links.add(full_url)

            print(f"  [발견된 링크] {len(found_links)}개")
            urls_to_visit.update(found_links)

        if knowledge_base:
            print(f"\n✅ 총 {len(knowledge_base)}개의 지식 덩어리를 {len(visited_urls)}개 페이지에서 최종 추출했습니다.")
        return knowledge_base

    def _setup_vector_db(self) -> chromadb.Collection:
        print("--- 🧠 벡터 DB 설정 및 지식 저장 시작 ---")
        chroma_client = chromadb.Client()
        collection_name = "chatbot_content_v5"
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = chroma_client.create_collection(name=collection_name)

        if not self.knowledge_base:
            print("⚠️ 저장할 지식이 없어 벡터 DB가 비어있습니다.")
            return collection

        collection.add(
            documents=[doc['content'] for doc in self.knowledge_base],
            metadatas=[doc['metadata'] for doc in self.knowledge_base],
            ids=[doc['id'] for doc in self.knowledge_base]
        )
        print(f"✅ 총 {collection.count()}개의 지식이 벡터 DB에 성공적으로 저장되었습니다.")
        return collection

    def _hybrid_retrieve(self, query: str, n_results: int = 5) -> str:
        """시맨틱 검색을 통해 관련 정보를 가져옵니다."""
        if self.db_collection.count() == 0:
            return ""

        semantic_results = self.db_collection.query(query_texts=[query], n_results=n_results)

        docs_with_metadata = []
        for i, doc in enumerate(semantic_results['documents'][0]):
            metadata = semantic_results['metadatas'][0][i]
            docs_with_metadata.append(f"[출처: {metadata.get('title', '알 수 없음')}]\n{doc}")

        return "\n\n".join(docs_with_metadata)

    def _generate_final_response(self, query: str, context: str, user_profile: Dict[str, Any],
                                 history: List[Dict[str, str]]) -> Dict[str, Any]:
        """단순하고 강력한 프롬프트를 사용하여 LLM에 최종 답변 생성을 요청합니다."""
        user_name = user_profile.get('name', '회원')
        functions_string = json.dumps(self.site_functions, indent=2, ensure_ascii=False)
        history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = f"""
        당신은 'DuoPet' 서비스의 유능하고 친절한 AI 비서입니다. 사용자 '{user_name}'님을 도와주세요.

        **지시사항:**

        1.  **정보 기반 답변:** 먼저, 아래 [참고 정보]를 사용하여 사용자의 [현재 질문]에 대한 답변을 찾으십시오.
            -   만약 관련 정보가 있다면, 그 정보를 바탕으로 친절하고 명확하게 답변하십시오.
            -   답변은 항상 '{user_name}님, '으로 시작하십시오.

        2.  **일반 지식 활용:**
            -   만약 [참고 정보]에 질문에 대한 답이 없다면, 그때는 당신의 일반 지식을 활용하여 최선을 다해 답변하십시오.
            -   "정보를 찾을 수 없습니다"라는 말 대신, 도움이 되는 일반적인 조언이나 정보를 제공하세요.

        3.  **기능 및 질문 제안:**
            -   답변 후, 사용자의 질문과 관련 있는 기능을 [사이트 기능 목록]에서 찾아 제안하십시오.
            -   사용자가 다음에 궁금해할 만한 **관련 후속 질문 3가지**를 예측하여 생성하십시오.

        4.  **출력 형식:** 최종 결과물은 반드시 아래 JSON 형식으로만 반환해야 합니다.

        **JSON 출력 형식:**
        {{
          "answer": "사용자에게 보여줄 텍스트 답변입니다.",
          "suggested_actions": ["가장 관련있는 함수의 name"],
          "predicted_questions": ["예상 질문 1", "예상 질문 2", "예상 질문 3"]
        }}

        ---
        [참고 정보]
        {context if context else "없음"}
        ---
        [이전 대화 내용]
        {history_string if history_string else "없음"}
        ---
        [사이트 기능 목록]
        {functions_string}
        ---
        [현재 질문]
        {query}
        ---

        JSON 출력:
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"당신은 '{self.site_url}' 웹사이트 전문 AI 어시스턴트이며, JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"🚨 OpenAI API 호출 중 오류가 발생했습니다: {e}")
            return {"answer": "죄송합니다, AI 모델과 통신하는 중에 문제가 발생했습니다.", "suggested_actions": [], "predicted_questions": []}

    def ask(self, query: str, user_profile: Dict[str, Any], history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        """메인 실행 함수"""
        context = self._hybrid_retrieve(query)
        print(f"\n[검색된 컨텍스트]\n---\n{context}\n---")

        response_json = self._generate_final_response(query, context, user_profile, history)

        if "suggested_actions" in response_json and isinstance(response_json["suggested_actions"], list):
            action_details = []
            valid_action_names = {func['name'] for func in self.site_functions}
            for action_name in response_json["suggested_actions"]:
                if action_name in valid_action_names:
                    for func in self.site_functions:
                        if func['name'] == action_name:
                            action_details.append({
                                "name": func['name'],
                                "description": func['description'],
                                "url": f"{self.base_url}{func['url']}"
                            })
            response_json["suggested_actions"] = action_details
        else:
            response_json["suggested_actions"] = []

        if "predicted_questions" not in response_json or not isinstance(response_json["predicted_questions"], list):
            response_json["predicted_questions"] = []

        return response_json