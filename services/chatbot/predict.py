import os
import json
import httpx
import functools
import time
import chromadb
import traceback
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI
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
from selenium.common.exceptions import TimeoutException

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


class RAGChatbot:
    def __init__(self, site_url: str, max_crawl_pages: int = 50):
        print("🤖 RAG 챗봇 초기화를 시작합니다...")
        self.site_url = site_url
        self.site_functions = [
            {"name": "notice_board", "description": "공지사항 확인하기", "url": "/notice"},
            {"name": "free_board", "description": "자유게시판 가기", "url": "/community/freeBoard"},
            {"name": "health_check", "description": "반려동물 건강 진단하기", "url": "/health/ai-diagnosis"},
            {"name": "behavior_analysis", "description": "이상행동 분석 서비스 보기", "url": "/health/ai-behavior"},
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

        self.predefined_questions = {
            "notice_board": [
                "최근 공지사항 3개만 알려줘",
                "서비스 점검 일정은 언제야?"
            ],
            "free_board": [
                "사람들이 가장 많이 본 글은 뭐야?",
                "강아지 자랑 게시판은 어디야?",
                "글을 쓰려면 어떻게 해야 해?"
            ],
            "health_check": [
                "우리 {pet_species} {pet_name}가(이) 자꾸 귀를 긁어",
                "우리 아이가 오늘따라 기운이 없어",
                "건강 진단 결과는 저장돼?"
            ],
            "behavior_analysis": [
                "강아지가 꼬리를 무는 이유는 뭐야?",
                "고양이가 밤에 너무 시끄럽게 울어",
                "분리불안 증상에 대해 알려줘"
            ],
            "qna": [
                "자주 묻는 질문은 뭐가 있어?",
                "결제 관련해서 질문하고 싶어",
                "내 질문에 대한 답변은 어디서 봐?"
            ],
            "default": [
                "{pet_age}살인 우리 {pet_name}에게 맞는 사료 추천해줘",
                "우리 {pet_species}가 좋아할 만한 장난감 있어?",
                "가장 인기 있는 서비스는 뭐야?"
            ]
        }
        self.base_url = f"{urlparse(self.site_url).scheme}://{urlparse(self.site_url).netloc}"
        self.max_crawl_pages = max_crawl_pages

        print("KeyBERT 모델을 로딩 중입니다...")
        self.kw_model = KeyBERT('paraphrase-multilingual-MiniLM-L12-v2')
        print("모델 로딩 완료.")

        self.chroma_db_path = os.environ.get("CHROMA_DB_PATH", "./api/chroma_data")
        self.db_collection = self._setup_vector_db()

        if self.db_collection.count() == 0:
            print("⚠️ 기존 지식 베이스가 비어있습니다. 사이트 크롤링을 시작합니다...")
            self.knowledge_base = self._create_kb_from_site()
            if not self.knowledge_base:
                raise RuntimeError("지식 베이스 생성에 실패했습니다. URL과 사이트 내용을 확인해주세요.")

            print(f"--- 🧠 크롤링된 지식 {len(self.knowledge_base)}개를 벡터 DB에 저장 중 ---")
            self.db_collection.add(
                documents=[doc['content'] for doc in self.knowledge_base],
                metadatas=[doc['metadata'] for doc in self.knowledge_base],
                ids=[doc['id'] for doc in self.knowledge_base]
            )
            print(f"✅ 총 {self.db_collection.count()}개의 지식이 벡터 DB에 성공적으로 저장되었습니다.")
        else:
            print(f"✅ 기존 벡터 DB에서 {self.db_collection.count()}개의 지식 로딩 완료. 크롤링을 건너뜀.")
            self.knowledge_base = []

    def _get_page_content(self, url: str) -> str:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(
            'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36')

        driver = None
        try:
            print("  [Selenium] ChromeDriver를 설정하는 중...")
            driver_path = ChromeDriverManager().install()
            service = ChromeService(executable_path=driver_path)

            driver = webdriver.Chrome(service=service, options=options)
            print(f"  [Selenium] '{url}' 페이지로 이동 중...")
            driver.get(url)

            wait = WebDriverWait(driver, 15)
            wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[class*='_container']"))
            )
            print("  [Selenium] 페이지의 주요 콘텐츠 컨테이너(_container) 로딩을 확인했습니다.")

            time.sleep(1)
            return driver.page_source
        except TimeoutException:
            print(f"  [Selenium 경고] '{url}' 페이지에서 '_container'를 시간 내에 찾지 못했습니다. 현재 페이지 소스를 기반으로 계속 진행합니다.")
            return driver.page_source if driver else ""
        except Exception as e:
            print(f"🚨 '{url}' 페이지 크롤링 중 심각한 오류 발생: {e}")
            traceback.print_exc()
            return ""
        finally:
            if driver:
                driver.quit()

    def _create_kb_from_site(self) -> List[Dict[str, Any]]:
        print(f"--- 🌐 사이트 전체 콘텐츠 추출 시작 (최대 {self.max_crawl_pages} 페이지) ---")
        urls_to_visit = {self.site_url}
        visited_urls = set()
        knowledge_base = []
        while urls_to_visit and len(visited_urls) < self.max_crawl_pages:
            current_url = urls_to_visit.pop()
            if current_url in visited_urls: continue
            print(f"\n[크롤링 시작] -> {current_url}")
            visited_urls.add(current_url)
            html_content = self._get_page_content(current_url)
            if not html_content:
                print("  [결과] 페이지 콘텐츠를 가져오지 못했습니다.")
                continue
            soup = BeautifulSoup(html_content, 'html.parser')
            page_title = soup.title.string.strip() if soup.title else '제목 없음'
            print(f"  [페이지 제목] {page_title}")

            for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
                tag.decompose()

            content_area = soup.select_one("div[class*='_container']") or soup.body

            if not content_area:
                print("  [결과] 주요 콘텐츠 영역을 찾지 못했습니다.")
                continue

            print(f"  [콘텐츠 영역 탐색] 선택된 영역: <{content_area.name} class='{' '.join(content_area.get('class', []))}'>")

            chunks_from_page = []

            for element in content_area.find_all(['h2', 'h3', 'p', 'td', 'li']):

                if isinstance(element, NavigableString): continue
                text = element.get_text(separator=' ', strip=True)
                if len(text.split()) > 3:
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
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(self.base_url, href)
                if full_url.startswith(self.base_url) and full_url not in visited_urls:
                    parsed_link = urlparse(full_url)
                    if not parsed_link.fragment and not (parsed_link.path.endswith(
                            ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.zip', '.pdf'))):
                        found_links.add(full_url)
            print(f"  [발견된 링크] {len(found_links)}개")
            urls_to_visit.update(found_links)
        if knowledge_base:
            print(f"\n✅ 총 {len(knowledge_base)}개의 지식 덩어리를 {len(visited_urls)}개 페이지에서 최종 추출했습니다.")
        return knowledge_base

    def _setup_vector_db(self) -> chromadb.Collection:
        chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        collection_name = "chatbot_content_v5"
        collection = chroma_client.get_or_create_collection(name=collection_name)
        print(f"✅ 기존 벡터 DB 컬렉션 '{collection_name}' 로드 또는 생성 성공.")
        return collection

    def resync_data_from_site(self):
        try:
            print("🔄 관리자 요청: 챗봇 데이터 전체 리프레시를 시작합니다.")
            current_count = self.db_collection.count()
            if current_count > 0:
                print(f"  - 기존 데이터 {current_count}개를 삭제합니다...")
                all_ids = self.db_collection.get(include=[])['ids']
                if all_ids: self.db_collection.delete(ids=all_ids)
                print(f"  - 기존 데이터 삭제 완료. 현재 카운트: {self.db_collection.count()}")
            print("  - 사이트 크롤링을 새로 시작합니다...")
            new_knowledge_base = self._create_kb_from_site()
            if not new_knowledge_base:
                print("🚨 리프레시 중 크롤링된 데이터가 없습니다. 작업을 중단합니다.")
                return
            print(f"  - 새로운 지식 {len(new_knowledge_base)}개를 벡터 DB에 저장합니다...")
            self.db_collection.add(
                documents=[doc['content'] for doc in new_knowledge_base],
                metadatas=[doc['metadata'] for doc in new_knowledge_base],
                ids=[doc['id'] for doc in new_knowledge_base]
            )
            final_count = self.db_collection.count()
            print(f"✅ 챗봇 데이터 리프레시 성공! 총 {final_count}개의 지식이 저장되었습니다.")
        except Exception as e:
            print(f"🚨 데이터 리프레시 중 심각한 오류 발생: {e}")
            traceback.print_exc()

    def _check_for_keyword_redirect(self, query: str) -> Dict[str, Any] | None:
        detected_actions = set()
        for keyword, actions in self.keyword_redirect_map.items():
            if keyword in query:
                for action in actions: detected_actions.add(action)
        if not detected_actions: return None
        action_details = []
        for action_name in detected_actions:
            for func in self.site_functions:
                if func['name'] == action_name:

                    action_details.append(
                        {"name": func['name'], "description": func['description'], "url": func['url']})
        if not action_details: return None
        return {"answer": "혹시 이런 기능들을 찾고 계신가요? 아래 버튼으로 빠르게 이동해 보세요.", "suggested_actions": action_details,
                "predicted_questions": []}

    def _hybrid_retrieve(self, query: str, n_results: int = 5, source_filter: str = None) -> str:
        if self.db_collection.count() == 0: return ""

        try:
            keywords = [keyword for keyword, score in self.kw_model.extract_keywords(query, top_n=5)]
            print(f"  [추출된 키워드] {keywords}")
        except Exception as e:
            print(f"🚨 KeyBERT 키워드 추출 중 오류 발생: {e}")
            keywords = []

        enhanced_query = query + " " + " ".join(keywords)
        print(f"  [강화된 검색어] {enhanced_query}")

        # 💡 개선: ChromaDB가 $like를 지원하지 않으므로, 더 많은 결과를 가져와 Python에서 직접 필터링합니다.
        # 필터링을 위해 충분한 후보군을 확보하도록 검색 결과 수를 늘립니다.
        query_n_results = 50 if source_filter else n_results

        query_params = {
            'query_texts': [enhanced_query],
            'n_results': query_n_results
        }

        # 💡 개선: where 절은 사용하지 않습니다.
        # if source_filter:
        #     where_clause = {"source": {"$like": f"%{source_filter}%"}} # 이 부분이 오류의 원인
        #     query_params['where'] = where_clause
        #     print(f"  [메타데이터 필터링 적용] source: {source_filter}")

        semantic_results = self.db_collection.query(**query_params)

        docs_with_metadata = []
        if semantic_results and semantic_results['documents']:
            documents = semantic_results['documents'][0]
            metadatas = semantic_results['metadatas'][0]

            for i, doc in enumerate(documents):
                metadata = metadatas[i]

                # 💡 개선: source_filter가 있는 경우, Python 코드 레벨에서 필터링을 수행합니다.
                if source_filter:
                    if source_filter in metadata.get('source', ''):
                        docs_with_metadata.append(f"[출처: {metadata.get('title', '알 수 없음')}]\n{doc}")
                else:
                    docs_with_metadata.append(f"[출처: {metadata.get('title', '알 수 없음')}]\n{doc}")

        # 최종적으로 요청된 개수(n_results)만큼만 반환합니다.
        print(f"  [필터링 후 컨텍스트] {len(docs_with_metadata)}개 문서 선택됨 (최대 {n_results}개 반환)")
        return "\n\n".join(docs_with_metadata[:n_results])

    def _generate_final_response(self, query: str, context: str, user_profile: Dict[str, Any],
                                 history: List[Dict[str, str]]) -> Dict[str, Any]:
        user_display_name = user_profile.get('nickname', user_profile.get('name', '회원'))
        functions_string = json.dumps(self.site_functions, indent=2, ensure_ascii=False)
        history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        user_profile_string_parts = []
        if user_profile.get('user_id'): user_profile_string_parts.append(f"사용자 ID: {user_profile['user_id']}")
        if user_profile.get('name'): user_profile_string_parts.append(f"이름: {user_profile['name']}")
        if user_profile.get('nickname'): user_profile_string_parts.append(f"닉네임: {user_profile['nickname']}")
        if user_profile.get('email'): user_profile_string_parts.append(f"이메일: {user_profile['email']}")
        if user_profile.get('member_since'): user_profile_string_parts.append(f"가입일: {user_profile['member_since']}")
        if user_profile.get('age'): user_profile_string_parts.append(f"나이: {user_profile['age']}세")
        if user_profile.get('gender'): user_profile_string_parts.append(f"성별: {user_profile['gender']}")
        if user_profile.get('phone'): user_profile_string_parts.append(f"전화번호: {user_profile['phone']}")
        if user_profile.get('address'): user_profile_string_parts.append(f"주소: {user_profile['address']}")
        if user_profile.get('pet_info'):
            for i, pet in enumerate(user_profile['pet_info']):
                pet_details = (f"반려동물 {i + 1}: 이름 {pet.get('name', '알 수 없음')}, 종 {pet.get('species', '알 수 없음')}")
                if pet.get('breed'): pet_details += f", 품종 {pet['breed']}"
                if pet.get('age'): pet_details += f", 나이 {pet['age']}"
                if pet.get('gender'): pet_details += f", 성별 {pet['gender']}"
                if pet.get('neutered') is not None: pet_details += f", 중성화 {pet['neutered']}"
                if pet.get('weight'): pet_details += f", 체중 {pet['weight']}"
                if pet.get('medical_history'): pet_details += f", 특이사항: {pet['medical_history']}"
                if pet.get('registration_date'): pet_details += f", 등록일: {pet['registration_date']}"
                user_profile_string_parts.append(pet_details)
        if user_profile.get('role'): user_profile_string_parts.append(f"사용자 시스템 역할: {user_profile['role']}")
        user_profile_string = "\n".join(user_profile_string_parts) if user_profile_string_parts else "없음"
        prompt = f"""
        당신은 'DuoPet' 서비스의 유능하고 친절한 AI 비서입니다. 사용자 '{user_display_name}'님을 도와주세요.
        답변은 항상 한국어로 제공하십시오.
        **지시사항:**
        1.  **정보 기반 답변:** 아래 [참고 정보]를 사용하여 사용자의 [현재 질문]에 대한 답변을 찾으십시오.
        2.  **개인화된 답변:** 아래 [사용자 프로필 정보]를 적극 활용하여 답변을 개인화하십시오. 특히 반려동물 관련 질문에는 해당 반려동물의 이름, 종 등을 언급하며 더 구체적으로 답변하십시오.
        3.  **일반 지식 활용:** [참고 정보]에 답이 없다면, 당신의 일반 지식을 활용하여 최선을 다해 답변하십시오.
        4.  **[필수] 후속 질문 제안:**
            -   답변이 끝난 후, 사용자가 다음에 궁금해할 만한 **관련 후속 질문 3가지를 반드시 예측하여 생성**해야 합니다.
            -   이 질문들은 사용자 프로필(특히 반려동물 정보)을 활용하여 개인화되어야 합니다.
            -   이 작업은 선택 사항이 아니며, **결과 JSON에 'predicted_questions' 키가 반드시 포함되어야 합니다.**
        5.  **[필수] 기능 제안 및 출력 형식:**
            -   답변과 관련 있는 기능을 [사이트 기능 목록]에서 찾아 제안하십시오.
            -   최종 결과물은 반드시 아래 JSON 형식으로만 반환해야 하며, **명시된 모든 키를 포함**해야 합니다.
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
            response = client.chat.completions.create(
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
            print(f"🚨 OpenAI API 호출 중 오류가 발생했습니다: {e}")
            return {"answer": "죄송합니다, AI 모델과 통신하는 중에 문제가 발생했습니다.", "suggested_actions": [], "predicted_questions": []}

    def ask(self, query: str, user_profile: Dict[str, Any], history: List[Dict[str, str]] = []) -> Dict[str, Any]:
        is_logged_in = user_profile and user_profile.get('user_id') not in [None, '0']
        user_display_name = user_profile.get('nickname', '고객')
        if is_logged_in and any(keyword in query for keyword in ["로그인", "가입"]):
            print(f"[로그인 상태 확인] '{user_display_name}'님은 이미 로그인 상태입니다. 확정된 답변을 즉시 반환합니다.")
            return {
                "answer": f"{user_display_name}님은 이미 로그인 상태입니다. 다른 도움이 필요하시면 편하게 말씀해주세요.",
                "suggested_actions": [
                    {"name": "free_board", "description": "자유게시판 가기", "url": f"{self.base_url}/board"},
                    {"name": "health_check", "description": "반려동물 건강 진단하기", "url": f"{self.base_url}/health-check"}
                ],
                "predicted_questions": ["내 정보는 어디서 확인해?", "우리 아이 건강 기록 보고 싶어", "자유게시판에 다른 사람들은 무슨 글을 썼어?"]
            }
        original_functions = self.site_functions
        if is_logged_in:
            print("[로그인 상태 확인] 추천 기능 목록에서 '로그인'을 임시로 제외합니다.")
            self.site_functions = [func for func in original_functions if func['name'] != 'login']
        try:
            try:
                spell_checker = SpellChecker()
                result_dict = spell_checker.check_spelling(query)
                corrected_query = result_dict['corrected_text']
                if query != corrected_query:
                    print(
                        f"\n[맞춤법 교정] 원본: '{query}' -> 교정: '{corrected_query}' (오류 {result_dict.get('error_count', 0)}개)")
                else:
                    print(f"\n[맞춤법 교정] 원본과 동일: '{query}'")
            except Exception as e:
                print(f"🚨 맞춤법 검사 중 오류 발생 (원본 질문 사용): '{e}'")
                corrected_query = query
            keyword_response = self._check_for_keyword_redirect(corrected_query)
            if keyword_response:
                print(f"\n[키워드 감지] '{corrected_query}'에 대한 빠른 응답 기능을 제공합니다.")
                return keyword_response
            source_filter = None
            if "공지" in corrected_query:
                source_filter = "/notice"
            elif "자유게시판" in corrected_query:
                source_filter = "/community/freeBoard"
            elif "질문" in corrected_query or "QnA" in corrected_query.upper():
                source_filter = "/qna"
            context = self._hybrid_retrieve(corrected_query, source_filter=source_filter)
            print(f"\n[검색된 컨텍스트]\n---\n{context if context else '관련 정보 없음'}\n---")
            response_json = self._generate_final_response(corrected_query, context, user_profile, history)
            if "suggested_actions" in response_json and isinstance(response_json["suggested_actions"], list):
                action_details = []
                valid_action_names = {func['name'] for func in self.site_functions}
                for action_name in response_json["suggested_actions"]:
                    if action_name in valid_action_names:
                        for func in self.site_functions:
                            if func['name'] == action_name:
                                action_details.append(
                                    {"name": func['name'], "description": func['description'], "url": func['url']})
                response_json["suggested_actions"] = action_details
            else:
                response_json["suggested_actions"] = []
            selected_questions = []
            if response_json.get("suggested_actions"):
                first_action_name = response_json["suggested_actions"][0]['name']
                selected_questions = self.predefined_questions.get(first_action_name,
                                                                   self.predefined_questions['default'])
            else:
                selected_questions = self.predefined_questions['default']
            final_questions = []
            pet = user_profile['pet_info'][0] if user_profile.get('pet_info') else None
            for q_template in selected_questions:
                if pet:
                    question = q_template.replace('{pet_name}', pet.get('name', '반려동물'))
                    question = question.replace('{pet_species}', pet.get('species', '반려동물'))
                    question = question.replace('{pet_age}', str(pet.get('age', 'N살')))
                    final_questions.append(question)
                else:
                    question = q_template.replace('{pet_name}', '반려동물')
                    question = question.replace('{pet_species}', '반려동물')
                    question = question.replace('{pet_age}', '우리 아이')
                    final_questions.append(question)
            response_json["predicted_questions"] = final_questions[:3]
            return response_json
        finally:
            self.site_functions = original_functions
            if is_logged_in:
                print("[요청 처리 완료] 기능 목록을 원래 상태로 복원합니다.")