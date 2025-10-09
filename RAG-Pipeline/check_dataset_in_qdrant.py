from qdrant_client import QdrantClient

# 기본 로컬 환경 (Docker로 실행)
client = QdrantClient(host="localhost", port=6333)

info = client.get_collection(collection_name="patent_db")
print(info)