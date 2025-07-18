

from typing import Dict, Any, Annotated
import oracledb
from datetime import date  # datetime.date 타입 처리를 위해 임포트
from fastapi import Depends
from starlette.concurrency import run_in_threadpool

from common.database import get_oracle_connection  # 기존의 동기 DB 연결을 가져옵니다.
from common.logger import get_logger

logger = get_logger(__name__)


async def get_user_profile_for_chatbot(
        user_id: str,
        conn: Annotated[oracledb.Connection, Depends(get_oracle_connection)]  # 동기 Connection을 주입받음
) -> Dict[str, Any]:
    """
    챗봇 서비스를 위해 사용자 ID를 기반으로 Oracle DB에서 사용자 프로필 정보를 가져오는 비동기 함수.
    기존 동기 DB 연결을 사용하여 비동기적으로 실행합니다.
    """
    # 기본 프로필 (비로그인 사용자 또는 DB 조회 실패 시 반환)
    user_profile = {
        "user_id": user_id,
        "name": "비회원",
        "nickname": None,
        "email": None,
        "member_since": None,
        "age": None,
        "gender": None,
        "phone": None,
        "address": None,
        "role": None,  # 역할 추가
        "pet_info": []
    }

    if user_id == '0':  # 비로그인 사용자 (프론트엔드에서 '0'으로 전달)
        logger.info(f"사용자 ID가 '0' (비회원)이므로 기본 프로필을 반환합니다.")
        return user_profile

    try:
        user_id_num = int(user_id)  # USER_ID는 NUMBER 타입이므로 int로 변환
    except ValueError:
        logger.error(f"Invalid user_id format received: '{user_id}'. Must be a number. Returning default profile.")
        return user_profile

    # 동기 DB 작업을 별도의 스레드 풀에서 실행하는 헬퍼 함수
    def fetch_data_sync(connection: oracledb.Connection, user_identifier: int) -> Dict[str, Any] | None:
        try:
            with connection.cursor() as cursor:
                # 1. USERS 테이블에서 사용자 정보 조회
                # 스키마에 맞춰 컬럼명 명시
                cursor.execute(
                    """
                    SELECT USER_ID,
                           USER_NAME,
                           NICKNAME,
                           PHONE,
                           AGE,
                           GENDER,
                           ADDRESS,
                           USER_EMAIL,
                           ROLE,
                           CREATED_AT
                    FROM USERS
                    WHERE USER_ID = :user_id_param
                    """,
                    user_id_param=user_identifier
                )
                user_row = cursor.fetchone()

                if not user_row:
                    return None  # 해당 USER_ID의 사용자를 찾지 못함

                # 컬럼 이름과 매핑: oracledb의 cursor.description을 활용하여 동적으로 매핑
                user_columns = [col[0] for col in cursor.description]
                user_data_raw = dict(zip(user_columns, user_row))

                # 데이터 가공 및 필드 매핑
                current_user_profile = {
                    "user_id": str(user_data_raw.get("USER_ID")),  # USER_ID는 NUMBER지만, 문자열로 통일
                    "name": user_data_raw.get("USER_NAME"),
                    "nickname": user_data_raw.get("NICKNAME"),
                    "email": user_data_raw.get("USER_EMAIL"),
                    "member_since": user_data_raw.get("CREATED_AT").strftime("%Y-%m-%d") if isinstance(
                        user_data_raw.get("CREATED_AT"), date) else None,
                    "age": user_data_raw.get("AGE"),
                    "gender": user_data_raw.get("GENDER"),
                    "phone": user_data_raw.get("PHONE"),
                    "address": user_data_raw.get("ADDRESS"),
                    "role": user_data_raw.get("ROLE"),  # ROLE 컬럼 추가
                    "pet_info": []  # 초기화
                }

                # 2. PETS 테이블에서 반려동물 정보 조회
                # USER_ID를 외래키로 사용하여 조회
                cursor.execute(
                    """
                    SELECT PET_NAME,
                           ANIMAL_TYPE,
                           BREED,
                           AGE,
                           GENDER,
                           NEUTERED,
                           WEIGHT,
                           REGISTRATION_DATE
                           
                    FROM PET
                    WHERE USER_ID = :owner_id_param
                    """,
                    owner_id_param=user_identifier
                )
                pet_rows = cursor.fetchall()

                pet_info_list = []
                # PETS 테이블의 컬럼 이름에 맞춰 동적 매핑 (MEDICAL_HISTORY는 스크린샷에 없으므로 일단 제외했습니다. 필요하면 추가하세요.)
                # 스크린샷에는 MEDICAL_HISTORY가 없었으나, 기존 코드에 존재하여 추가했습니다.
                # 만약 실제 DB에 없다면 이 부분을 제거하거나 주석 처리해야 합니다.
                pet_columns = [col[0] for col in cursor.description]

                for pet_row in pet_rows:
                    pet_data_raw = dict(zip(pet_columns, pet_row))

                    pet_info_list.append({
                        "name": pet_data_raw.get("PET_NAME"),
                        "species": pet_data_raw.get("ANIMAL_TYPE"),
                        "breed": pet_data_raw.get("BREED"),
                        "age": f"{int(pet_data_raw.get('AGE'))}세" if pet_data_raw.get('AGE') is not None else None,
                        # NUMBER(3,0) 이므로 int로 변환
                        "gender": pet_data_raw.get("GENDER"),
                        "neutered": pet_data_raw.get("NEUTERED"),
                        "weight": float(pet_data_raw.get("WEIGHT")) if pet_data_raw.get('WEIGHT') is not None else None,
                        # FLOAT 이므로 float로 변환
                        "registration_date": pet_data_raw.get("REGISTRATION_DATE").strftime("%Y-%m-%d") if isinstance(
                            pet_data_raw.get("REGISTRATION_DATE"), date) else None,
                        "medical_history": pet_data_raw.get("MEDICAL_HISTORY")  # 스키마에 없으면 None 반환될 것
                    })

                current_user_profile["pet_info"] = pet_info_list
                return current_user_profile

        except Exception as e:
            logger.error(f"Error fetching user/pet data from DB for user '{user_identifier}': {e}", exc_info=True)
            return None  # 오류 발생 시 None 반환

    # run_in_threadpool을 사용하여 동기 DB 작업을 비동기적으로 실행
    profile_data = await run_in_threadpool(fetch_data_sync, conn, user_id_num)

    if profile_data:
        return profile_data
    else:
        logger.warning(
            f"User with ID '{user_id}' not found or error occurred during fetching. Returning default profile.")
        return user_profile