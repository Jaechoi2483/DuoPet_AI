# --------------------------------------------------------------------------
# 파일명: services/chatbot/predict.py
# 설명: 답변 생성 및 예상 질문 생성 로직 개선 (LLM 프롬프트 최적화 및 비동기 처리)
# --------------------------------------------------------------------------
import os
import json
import httpx
import functools
# import time # 사용되지 않으므로 제거
import chromadb
from bs4 import BeautifulSoup, NavigableString
from openai import AsyncOpenAI  # 💡 비동기 클라이언트로 변경
from keybert import KeyBERT
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Dict, Any
from kospellcheck import SpellChecker
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from common.logger import get_logger  # 로깅을 위해 추가

# logger 객체를 모듈 전역에서 사용하도록 정의합니다.
logger = get_logger(__name__)

global_chat_cache = {}

# --- 환경 변수 로드 및 라이브러리 동작 수정 ---
load_dotenv()

# httpx.Client 동기 클라이언트 프록시 패치
original_init = httpx.Client.__init__


@functools.wraps(original_init)
def patched_init(self, *args, **kwargs):
    if 'proxies' in kwargs:
        del kwargs['proxies']
    original_init(self, *args, **kwargs)


httpx.Client.__init__ = patched_init

# httpx.AsyncClient 비동기 클라이언트 프록시 패치 (AsyncOpenAI가 사용하므로 다시 추가)
original_async_init = httpx.AsyncClient.__init__


@functools.wraps(original_async_init)
def patched_async_init(self, *args, **kwargs):
    if 'proxies' in kwargs:
        del kwargs['proxies']
    original_async_init(self, *args, **kwargs)


httpx.AsyncClient.__init__ = patched_async_init

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("🚨 OPENAI_API_KEY가 .env 파일에 설정되지 않았거나 비어 있습니다.")

try:
    # 💡 비동기 요청을 위해 AsyncOpenAI 클라이언트를 사용합니다. (현재 코드를 따름)
    client = AsyncOpenAI(api_key=api_key)
except Exception as e:
    raise RuntimeError(f"🚨 OpenAI 클라이언트 초기화 실패: {e}")


# --- RAG 챗봇 클래스 정의 ---
class RAGChatbot:
    def __init__(self, site_url: str, max_crawl_pages: int = 10):
        logger.info("🤖 RAG 챗봇 초기화를 시작합니다...")  # print -> logger.info
        self.site_url = site_url
        self.site_functions = [
            {"name": "notice_board", "description": "공지사항 확인하기", "url": "/notice"},
            {"name": "free_board", "description": "자유게시판 가기", "url": "/board"},
            {"name": "health_check", "description": "반려동물 건강 진단하기", "url": "/health-check"},
            {"name": "behavior_analysis", "description": "이상행동 분석 서비스 보기", "url": "/behavior-analysis"},
            {"name": "video_recommend", "description": "추천 영상 보러가기", "url": "/recommendations"},
            {"name": "qna", "description": "qna", "url": "/qna"},
            {"name": "login", "description": "로그인", "url": "/login"}
        ]

        self.keyword_redirect_map = {
            "궁금": ["qna", "faq", "free_board"],
            "질문": ["qna", "faq", "free_board"],
            "아파": ["health_check", "behavior_analysis"],
            "진단": ["health_check", "behavior_analysis"],
            "방법": ["notice_board", "faq"],
            "로그인": ["login"],
            "가입": ["login"],
            "심심": ["video_recommend", "free_board"]
        }

        self.base_url = f"{urlparse(self.site_url).scheme}://{urlparse(self.site_url).netloc}"
        self.max_crawl_pages = max_crawl_pages

        self.excluded_paths = [


            '/adoption'

        ]

        logger.info("KeyBERT 모델을 로딩 중입니다...")  # print -> logger.info
        self.kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("모델 로딩 완료.")  # print -> logger.info

        self.chroma_db_path = os.environ.get("CHROMA_DB_PATH", "./chroma_data")
        self.db_collection = self._setup_vector_db()

        if self.db_collection.count() == 0:
            logger.warning("⚠️ 기존 지식 베이스가 비어있습니다. 사이트 크롤링을 시작합니다...")  # print -> logger.warning
            self.knowledge_base = self._create_kb_from_site()
            if not self.knowledge_base:
                raise RuntimeError("지식 베이스 생성에 실패했습니다. URL과 사이트 내용을 확인해주세요.")

            logger.info(f"--- 🧠 크롤링된 지식 {len(self.knowledge_base)}개를 벡터 DB에 저장 중 ---")  # print -> logger.info
            self.db_collection.add(
                documents=[doc['content'] for doc in self.knowledge_base],
                metadatas=[doc['metadata'] for doc in self.knowledge_base],
                ids=[doc['id'] for doc in self.knowledge_base]
            )
            logger.info(f"✅ 총 {self.db_collection.count()}개의 지식이 벡터 DB에 성공적으로 저장되었습니다.")  # print -> logger.info
        else:
            logger.info(f"✅ 기존 벡터 DB에서 {self.db_collection.count()}개의 지식 로딩 완료. 크롤링을 건너뜀.")  # print -> logger.info
            self.knowledge_base = []

    def _get_page_content(self, url: str) -> str:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        options.add_argument('--window-size=1920,1080')

        driver = None
        try:
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)

            logger.info(f" [Selenium] '{url}' 페이지로 이동 중...")  # print -> logger.info
            driver.get(url)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                logger.info(" [Selenium] 페이지 로드 완료 대기 성공.")  # print -> logger.info
            except Exception as wait_e:
                logger.warning(f" [Selenium] 페이지 로드 대기 중 타임아웃 또는 오류 발생: {wait_e}")  # print -> logger.warning

            html_content = driver.page_source

            # 디버깅용 HTML 출력 및 파일 저장 (필요 없다면 제거)
            logger.debug(f"\n--- 가져온 HTML 콘텐츠 (상위 500자) ---\n{html_content[:500]}...\n---")  # print -> logger.debug
            # with open("crawled_page_content.html", "w", encoding="utf-8") as f:
            #     f.write(html_content)
            # logger.debug(f"💡 가져온 HTML 콘텐츠를 'crawled_page_content.html' 파일에 저장했습니다.") # print -> logger.debug

            return html_content
        except Exception as e:
            logger.error(f"🚨 '{url}' 페이지 크롤링 중 오류 발생: {e}", exc_info=True)  # print -> logger.error
            return ""
        finally:
            if driver:
                driver.quit()

        # RAGChatbot 클래스 내부에 다른 메소드들과 함께 위치해야 합니다.

    def _create_kb_from_site(self) -> List[Dict[str, Any]]:
        logger.info(f"--- 🌐 사이트 전체 콘텐츠 추출 시작 (최대 {self.max_crawl_pages} 페이지) ---")

        urls_to_visit = {self.site_url}
        visited_urls = set()
        knowledge_base = []

        while urls_to_visit and len(visited_urls) < self.max_crawl_pages:
            current_url = urls_to_visit.pop()
            if current_url in visited_urls:
                continue

            logger.info(f"\n[크롤링 시작] -> {current_url}")
            visited_urls.add(current_url)
            html_content = self._get_page_content(current_url)
            if not html_content:
                logger.warning(" [결과] 페이지 콘텐츠를 가져오지 못했습니다.")
                continue

            soup = BeautifulSoup(html_content, 'html.parser')
            page_title = soup.title.string.strip() if soup.title else '제목 없음'
            logger.info(f" [페이지 제목] {page_title}")

            content_area = soup.find('main') or soup.find('article') or soup.find('body')
            if not content_area:
                logger.warning(" [결과] 주요 콘텐츠 영역을 찾지 못했습니다. 전체 body에서 추출 시도.")
                content_area = soup.body

            chunks_from_page = []
            for element in content_area.find_all(
                    ['h1', 'h2', 'h3', 'h4', 'p', 'div', 'li', 'span', 'a', 'strong', 'em', 'dd', 'dt'],
                    recursive=True
            ):
                if isinstance(element, NavigableString): continue
                text = element.get_text(separator=' ', strip=True)
                if len(text) > 15 and '\n' not in text and 'function' not in text.lower() and 'var' not in text.lower():
                    chunks_from_page.append(text)

            unique_chunks = list(dict.fromkeys(chunks_from_page))

            for i, chunk in enumerate(unique_chunks):
                tags_list = []
                try:
                    tags_list = [kw for kw, score in self.kw_model.extract_keywords(chunk, top_n=3)]
                except Exception as e:
                    logger.warning(f"KeyBERT 태그 추출 중 오류 발생 (텍스트: '{chunk[:30]}...'): {e}")
                    tags_list = []

                tags_text = " ".join(tags_list)
                content_with_tags = f"{chunk}\n\n[TAGS: {tags_text}]"

                knowledge_base.append({
                    "id": f"{urlparse(current_url).path.replace('/', '_')}_{i}",
                    "content": content_with_tags,
                    "metadata": {"source": current_url, "title": page_title}
                })

            logger.info(f" [추출된 정보] {len(unique_chunks)}개의 텍스트 조각")

            found_links = set()
            for link in content_area.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.base_url, href)
                parsed_link = urlparse(full_url)
                if (full_url.startswith(self.base_url) and
                        full_url not in visited_urls and
                        parsed_link.path not in self.excluded_paths and  # 👈 제외 목록 확인
                        not parsed_link.fragment and
                        not (parsed_link.path.endswith(
                            ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.xml', '.txt', '.pdf')))):
                    found_links.add(full_url)

            logger.info(f" [발견된 링크] {len(found_links)}개")
            urls_to_visit.update(found_links)

        if knowledge_base:
            logger.info(
                f"\n✅ 총 {len(knowledge_base)}개의 지식 덩어리를 {len(visited_urls)}개 페이지에서 최종 추출했습니다.")
        return knowledge_base

    def _setup_vector_db(self) -> chromadb.Collection:
        chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection_name = "chatbot_content_v5"

        try:
            collection = chroma_client.get_or_create_collection(name=collection_name)
            logger.info(f"✅ 기존 벡터 DB 컬렉션 '{collection_name}' 로드 또는 생성 성공.")  # print -> logger.info
        except Exception as e:
            logger.warning(f"⚠️ 벡터 DB 컬렉션 '{collection_name}' 로딩 중 오류 발생. 새로 생성합니다. 오류: {e}")  # print -> logger.warning
            collection = chroma_client.create_collection(name=collection_name)
        return collection

    def resync_data_from_site(self):
        try:
            logger.info("🔄 관리자 요청: 챗봇 데이터 전체 리프레시를 시작합니다.")  # print -> logger.info

            current_count = self.db_collection.count()
            if current_count > 0:
                logger.info(f" - 기존 데이터 {current_count}개를 삭제합니다...")  # print -> logger.info
                all_ids = self.db_collection.get(include=[])['ids']
                if all_ids:
                    self.db_collection.delete(ids=all_ids)
                logger.info(f" - 기존 데이터 삭제 완료. 현재 카운트: {self.db_collection.count()}")  # print -> logger.info

            logger.info(" - 사이트 크롤링을 새로 시작합니다...")  # print -> logger.info
            new_knowledge_base = self._create_kb_from_site()
            if not new_knowledge_base:
                logger.error("🚨 리프레시 중 크롤링된 데이터가 없습니다. 작업을 중단합니다.")  # print -> logger.error
                return

            logger.info(f" - 새로운 지식 {len(new_knowledge_base)}개를 벡터 DB에 저장합니다...")  # print -> logger.info
            self.db_collection.add(
                documents=[doc['content'] for doc in new_knowledge_base],
                metadatas=[doc['metadata'] for doc in new_knowledge_base],
                ids=[doc['id'] for doc in new_knowledge_base]
            )

            final_count = self.db_collection.count()
            logger.info(f"✅ 챗봇 데이터 리프레시 성공! 총 {final_count}개의 지식이 저장되었습니다.")  # print -> logger.info

        except Exception as e:
            logger.error(f"🚨 데이터 리프레시 중 심각한 오류 발생: {e}", exc_info=True)  # print -> logger.error

    def _check_for_keyword_redirect(self, query: str) -> Dict[str, Any] | None:
        detected_actions = set()
        for keyword, actions in self.keyword_redirect_map.items():
            if keyword in query:
                for action in actions:
                    detected_actions.add(action)
        if not detected_actions:
            return None
        return {
            "answer": "혹시 이런 기능들을 찾고 계신가요? 아래 버튼으로 빠르게 이동해 보세요.",
            "suggested_actions": list(detected_actions),
            "predicted_questions": []
        }

    def _hybrid_retrieve(self, query: str, n_results: int = 5) -> str:
        """
        KeyBERT로 키워드를 추출하고, 메타데이터 필터링과 시맨틱 검색을 함께 수행합니다.
        """
        if self.db_collection.count() == 0:
            return ""

        try:
            # 질문에서 키워드 추출 (필터링에 사용)
            keywords = [keyword for keyword, score in self.kw_model.extract_keywords(query, top_n=3)]
            logger.info(f" KeyBERT 추출 키워드: {keywords}")
        except Exception as e:
            logger.error(f"🚨 KeyBERT 키워드 추출 중 오류 발생: {e}", exc_info=True)
            keywords = []

        # 👇 [추가] 추출된 키워드를 기반으로 ChromaDB where 필터를 생성합니다.
        where_document_filter = None
        if keywords:
            # 문서 내용(content)에 키워드가 포함되어 있는지 검색($contains)
            where_document_filter = {
                "$or": [{"$contains": keyword} for keyword in keywords]
            }

        enhanced_query = query + " " + " ".join(keywords)
        logger.info(f" 강화된 검색어: {enhanced_query}")

        # 👇 [수정] db_collection.query 호출 시 where 필터를 추가합니다.
        semantic_results = self.db_collection.query(
            query_texts=[enhanced_query],
            n_results=n_results,
            where_document=where_document_filter
        )

        docs_with_metadata = []
        if semantic_results and semantic_results['documents']:
            for i, doc in enumerate(semantic_results['documents'][0]):
                metadata = semantic_results['metadatas'][0][i]
                clean_doc = doc.split("\n\n[TAGS:")[0]
                docs_with_metadata.append(f"[출처: {metadata.get('title', '알 수 없음')}]\n{doc}")
        final_context = "\n\n".join(docs_with_metadata)
        logger.info(f"ChromaDB 검색 결과 (RAG 컨텍스트):\n---\n{final_context if final_context else '검색된 결과 없음.'}\n---")
        return final_context

    async def _generate_answer_and_actions(self, query: str, context: str, user_profile: Dict[str, Any],
                                           history: List[Dict[str, str]]) -> Dict[str, Any]:
        """답변 및 추천 액션, 예상 질문을 함께 생성하는 함수"""
        user_display_name = user_profile.get('nickname', user_profile.get('name', '비회원'))

        # 💡 사용자 프로필 정보를 프롬프트에 더 상세히 포함시키기
        user_profile_string_parts = []
        if user_profile.get('user_id'):
            user_profile_string_parts.append(f"사용자 ID: {user_profile['user_id']}")
        if user_profile.get('name'):
            user_profile_string_parts.append(f"이름: {user_profile['name']}")
        if user_profile.get('nickname'):
            user_profile_string_parts.append(f"닉네임: {user_profile['nickname']}")
        if user_profile.get('email'):
            user_profile_string_parts.append(f"이메일: {user_profile['email']}")
        if user_profile.get('member_since'):
            user_profile_string_parts.append(f"가입일: {user_profile['member_since']}")
        if user_profile.get('age'):
            user_profile_string_parts.append(f"나이: {user_profile['age']}세")
        if user_profile.get('gender'):
            user_profile_string_parts.append(f"성별: {user_profile['gender']}")
        if user_profile.get('phone'):
            user_profile_string_parts.append(f"전화번호: {user_profile['phone']}")
        if user_profile.get('address'):
            user_profile_string_parts.append(f"주소: {user_profile['address']}")

        if user_profile.get('pet_info'):
            for i, pet in enumerate(user_profile['pet_info']):
                pet_details = (
                    f"반려동물 {i + 1}: "
                    f"이름 {pet.get('name', '알 수 없음')}, "
                    f"종류 {pet.get('species', '알 수 없음')}, "
                    f"품종 {pet.get('breed', '알 수 없음')}, "
                    f"나이 {pet.get('age', '알 수 없음')}세, "
                    f"성별 {pet.get('gender', '알 수 없음')}, "
                    f"중성화 여부: {pet.get('neutered', '알 수 없음')}, "
                    f"체중: {pet.get('weight', '알 수 없음')}kg"  # 'kg' 단위 추가
                )
                if pet.get('medical_history'): pet_details += f", 특이사항: {pet['medical_history']}"
                if pet.get('registration_date'): pet_details += f", 등록일: {pet['registration_date']}"
                user_profile_string_parts.append(pet_details)
        else:
            user_profile_string_parts.append("반려동물 정보: 없음")

        if user_profile.get('role'):
            user_profile_string_parts.append(f"사용자 시스템 역할: {user_profile['role']}")

        user_profile_string = "\n".join(user_profile_string_parts) if user_profile_string_parts else "없음"

        functions_string = json.dumps(self.site_functions, indent=2, ensure_ascii=False)
        history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        prompt = f"""
        당신은 'DuoPet' 서비스의 유능하고 친절한 AI 비서입니다. 사용자 '{user_display_name}'님을 도와주세요.
        답변은 항상 한국어로 제공하십시오.

        **지시사항:**

        1.  **정보 기반 답변:** 아래 [참고 정보]를 사용하여 사용자의 [현재 질문]에 대한 답변을 찾으십시오.
            -   만약 관련 정보가 있다면, 그 정보를 바탕으로 친절하고 명확하게 답변하십시오.
            -   참고 정보에 반려동물 이름이 있다면, **질문과 관련될 경우 반드시 답변에 정확히 포함하여 구체적으로 언급**해야 합니다.

        2.  **개인화된 답변:**
            -   **아래 [사용자 프로필 정보]를 적극 활용하여 답변을 개인화하십시오.**
            -   특히 반려동물 관련 질문에는 해당 반려동물의 이름, 종, 나이 등을 언급하며 더 구체적으로 답변하십시오.
            -   사용자의 과거 활동이나 선호도에 기반하여 관련성 높은 정보를 제공하거나 기능을 제안하십시오.
            -   **사용자 프로필의 '사용자 시스템 역할' 정보(예: 관리자)를 답변에 직접적인 호칭으로 사용하지 마십시오.** 오직 사용자 '{user_display_name}'님만을 호칭으로 사용하십시오.

        3.  **일반 지식 활용:**
            -   만약 [참고 정보]에 질문에 대한 답이 없다면, 그때는 당신의 일반 지식을 활용하여 최선을 다해 답변하십시오.
            -   "정보를 찾을 수 없습니다"라는 말 대신, 도움이 되는 일반적인 조언이나 정보를 제공하세요.

        4.  **기능 및 질문 제안:**
            -   답변 후, 사용자의 질문과 관련 있는 기능을 [사이트 기능 목록]에서 찾아 제안하십시오.
            -   사용자가 다음에 궁금해할 만한 **관련 후속 질문 3가지**를 예측하여 생성하십시오.
                - 예상 질문 생성 시에도 사용자 프로필(특히 반려동물 정보)을 활용하여 개인화된 질문을 제안하십시오.

        5.  **출력 형식:** 최종 결과물은 반드시 아래 JSON 형식으로만 반환해야 합니다.

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
        [사용자 프로필 정보]
        {user_profile_string}
        ---
        [현재 질문]
        {query}
        ---

        JSON 출력:
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system",
                     "content": f"당신은 '{self.site_url}' 웹사이트 전문 AI 어시스턴트이며, 사용자 프로필 정보를 활용하여 개인화된 JSON 형식으로만 응답합니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"🚨 OpenAI API 호출 중 오류가 발생했습니다: {e}", exc_info=True)  # print -> logger.error
            return {"answer": "죄송합니다, AI 모델과 통신하는 중에 문제가 발생했습니다.", "suggested_actions": [], "predicted_questions": []}

    async def _is_context_relevant(self, query: str, context: str) -> bool:
        """RAG로 검색된 정보가 질문과 관련이 있는지 AI를 통해 확인합니다."""
        # 컨텍스트가 비어있으면 확인할 필요 없이 False를 반환합니다.
        if not context:
            return False

        # AI에게 보낼 간단하고 명확한 프롬프트
        prompt = f"""
        사용자 질문: "{query}"

        검색된 정보: "{context}"

        위 사용자 질문과 검색된 정보가 서로 의미적으로 관련이 있습니까? 오직 '네' 또는 '아니오' 한 단어로만 대답하십시오.
        """
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an assistant that determines if a context is relevant to a query. Answer only with '네' or '아니오'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # 일관된 답변을 위해 0으로 설정
                max_tokens=5  # '네' 또는 '아니오'만 받기 위해 토큰 수 제한
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"RAG 관련성 검사 결과: '{answer}'")
            # 응답에 '네'가 포함되어 있으면 True를 반환합니다.
            return "네" in answer
        except Exception as e:
            logger.error(f"RAG 관련성 검사 중 오류 발생: {e}")
            # 오류 발생 시, 안전하게 관련 없는 것으로 간주합니다.
            return False

    # ask 함수 전체를 아래 내용으로 교체합니다.

    async def ask(self, query: str, user_profile: Dict[str, Any], history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        """메인 실행 함수 (백엔드 필터링 제거, 프론트엔드 제어로 변경)"""

        # 맞춤법 검사 및 캐시 확인 (기존과 동일)
        try:
            spell_checker = SpellChecker()
            corrected_query = spell_checker.check_spelling(query)['corrected_text']
            if query != corrected_query: logger.info(f"[맞춤법 교정] '{query}' -> '{corrected_query}'")
        except Exception as e:
            logger.error(f"🚨 맞춤법 검사 중 오류: {e}")
            corrected_query = query

        if corrected_query in global_chat_cache:
            logger.info(f"캐시 히트(Cache Hit)!: '{corrected_query}'")
            return global_chat_cache[corrected_query]

        # 키워드 또는 AI를 통해 응답 생성 (기존과 동일)
        response_json = self._check_for_keyword_redirect(corrected_query)
        if not response_json:
            context = self._hybrid_retrieve(corrected_query)
            response_json = await self._generate_answer_and_actions(corrected_query, context, user_profile, history)

        # 👇 [수정] 로그인 상태 필터링 로직을 모두 제거하고,
        # 단순히 모든 추천 액션의 상세 정보만 생성합니다.
        action_details = []
        if "suggested_actions" in response_json and isinstance(response_json["suggested_actions"], list):
            valid_action_names = {func['name'] for func in self.site_functions}
            for name in response_json["suggested_actions"]:
                if name.strip() in valid_action_names:
                    for func in self.site_functions:
                        if func['name'] == name.strip():
                            action_details.append({"name": func['name'], "description": func['description'],
                                                   "url": f"{self.base_url}{func['url']}"})
                            break

        response_json["suggested_actions"] = action_details

        if "predicted_questions" not in response_json:
            response_json["predicted_questions"] = []

        # 캐시 저장 및 반환
        global_chat_cache[corrected_query] = response_json
        return response_json