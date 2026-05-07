"""analyze.py — Bin 불량 분석 시스템 CLI 진입점."""
import asyncio
import sys
import logging
from pathlib import Path

# .env.onprem 로드
try:
    from dotenv import load_dotenv
    load_dotenv(".env.onprem")
except ImportError:
    pass

from src.config import load_config
from src.llm_client import llm_complete
from src.knowledge import KnowledgeBase
from src.query_analyzer import analyze_query
from src.plan_generator import generate_plan
from src.step_executor import StepExecutor
from src.db_client import DatabaseClient
from src.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # 1) 설정 로드
    cfg = load_config("config.yaml")
    print("[1/6] 설정 로드 완료")

    # 2) DB 연결 확인
    db = DatabaseClient(
        host=cfg.database.host,
        port=cfg.database.port,
        database=cfg.database.database,
        username=cfg.database.username,
        password=cfg.database.password,
    )
    if not db.test_connection():
        print("DB 연결 실패. config.yaml과 .env.onprem을 확인하세요.")
        sys.exit(1)
    print("[2/6] DB 연결 확인 완료")

    # 3) LightRAG 지식 베이스 초기화
    from lightrag_onprem_demo import onprem_llm_complete, LLM_MODEL

    kb = KnowledgeBase(
        working_dir=cfg.lightrag.working_dir,
        embed_model_path=cfg.lightrag.embed_model_path,
        llm_model_func=onprem_llm_complete,
        llm_model_name=LLM_MODEL,
    )

    rag_index = Path(cfg.lightrag.working_dir) / "vdb_chunks.json"
    if not rag_index.exists():
        print("[3/6] 도메인 지식 인덱싱 시작...")
        await kb.initialize()
        await kb.index_documents(cfg.lightrag.domain_docs_dir)
        print("[3/6] 인덱싱 완료")
    else:
        print("[3/6] 기존 인덱스 사용")
        await kb.initialize()

    # 4) 사용자 입력 루프
    print("\n" + "=" * 60)
    print("  Bin 불량 분석 시스템 (Phase 1: 승인 모드)")
    print("  종료: Ctrl+C 또는 'quit' 입력")
    print("=" * 60)

    while True:
        try:
            query = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        try:
            # 4a) 쿼리 분석
            print("\n[4/6] 쿼리 분석 중...")
            params = await analyze_query(query)
            print(f"  추출 파라미터: {params}")

            # 4b) 도메인 지식 검색
            print("[5/6] 도메인 지식 검색 중...")
            knowledge_query = f"{params.get('process', '')} {params.get('issue', '')} Bin{params.get('bin_code', '')} 분석 플로우 테이블 스키마"
            knowledge = await kb.query(knowledge_query.strip(), mode="hybrid")

            schema_query = f"테이블 스키마 조인 관계 {' '.join(params.get('keywords', []))}"
            schema_knowledge = await kb.query(schema_query.strip(), mode="local")

            # 4c) 분석 계획 수립
            print("[6/6] 분석 계획 수립 중...")
            plan = await generate_plan(params, knowledge)
            print(f"\n[분석 계획] {plan.get('plan_name', '분석')}")
            for step in plan.get("steps", []):
                print(f"  Step {step['step_number']}: {step['name']}")

            proceed = input("\n진행하시겠습니까? (y/n): ").strip().lower()
            if proceed != "y":
                continue

            # 4d) 단계별 실행
            executor = StepExecutor(db=db, safety=cfg.safety, execution=cfg.execution)
            for step in plan.get("steps", []):
                await executor.execute_step(step, schema_knowledge)

            # 4e) 최종 리포트
            print("\n" + "=" * 60)
            print("  최종 분석 리포트")
            print("=" * 60)
            report = await generate_report(executor.get_all_results())
            print(report)

        except KeyboardInterrupt:
            print("\n분석 중단")
        except Exception as e:
            logger.error(f"분석 오류: {e}", exc_info=True)
            print(f"\n오류 발생: {e}")

    await kb.finalize()
    print("\n종료.")


if __name__ == "__main__":
    asyncio.run(main())
