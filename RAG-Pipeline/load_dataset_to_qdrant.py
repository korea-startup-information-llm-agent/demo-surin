import json
import ast

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

# 임베딩할 데이터 가져오기
with open("RAG-Pipeline/dataset/for_embedding.json", "r", encoding="utf-8") as f:
    total_data = json.load(f)

embedding_model =  SentenceTransformer("dragonkue/BGE-m3-ko")

# 기본 로컬 환경 (Docker로 실행)
client = QdrantClient(host="localhost", port=6333)

# 현재 존재하는 모든 컬렉션 목록 확인하기
# collections = client.get_collections()
# print(collections)

# 새로운 컬렉션 만들기
client.recreate_collection(
    collection_name="patent_db",
    vectors_config={
        "size": 1024, "distance": "Cosine",
    }
)

# 벡터 데이터 삽입
for i, data in enumerate(total_data, 1):
    id = i
    document_id = data["document_id"]
    application_date = f'{data["application_year"]}-{data["application_month"]}-{data["application_day"]}'
    invention_title = data["invention_title"]
    applicant_name = data["applicant_name"]
    application_number = data["application_number"]
    claims = data["claims"]
    abstract = data["abstract"]
    keyword = data["keyword"]

    # 키워드 리스트 처리하기
    keyword = data["keyword"]
    if isinstance(keyword, list):
        keyword = keyword
    elif isinstance(keyword, str):
        try:
            keyword = json.loads(keyword)
        except Exception as e:
            try:
                keyword = ast.literal_eval(keyword)
            except Exception as e:
                keyword = None
    else:
        keyword = None

    embedding_data = f"[발명의명칭] {invention_title} [요약] {abstract} [주요 키워드] {', '.join(keyword)}"
    embeddings = embedding_model.encode([embedding_data])

    client.upsert(
        collection_name="patent_db",
        points=[
            {
                "id": id,
                "vector": embeddings[0].tolist(),
                # 메타데이터
                "payload": {
                    "document_id": document_id,
                    "invention_title": invention_title,
                    "application_number": application_number,
                    "applicant_name": applicant_name,
                    "application_date": application_date,
                    "claims": claims,
                }
            }
        ]
    )