import json
import psycopg2
# import numpy as np

from sentence_transformers import SentenceTransformer

# 데이터베이스 연결
db = psycopg2.connect(host="localhost", database="postgres", user="ssr", port=5432)
print("Database connected successfully")

# 데이터 조작을 위한 객체 생성
cursor = db.cursor()

embedding_model = SentenceTransformer("dragonkue/BGE-m3-ko")

# json 파일 읽기
with open("RAG-Pipeline/dataset/for_embedding.json", "r", encoding="utf-8") as f:
    total_data = json.load(f)

print(f"Total data: {len(total_data)}")

# 임베딩 데이터 생성
embedding_data = []

for data in total_data:
    
    # 필요한 필드 추출하기
    document_id = data["document_id"]
    invention_info = data["invention_info"]    # 임베딩할 내용

    # 이미 들어있는지 확인
    cursor.execute("SELECT EXISTS (SELECT 1 FROM patents WHERE document_id=%s)", (document_id,))
    if cursor.fetchone()[0]:
        print(f"Patents with document_id {document_id} already exists in the database.")
        
        embeddings = embedding_model.encode([invention_info])

        query = "UPDATE patents SET embeddings=%s WHERE document_id=%s"
        input_data = (embeddings[0].tolist(), document_id)
        cursor.execute(query, input_data)
        db.commit()
        print(f"Patents with document_id {document_id} updated embeddings.")

cursor.close()
db.close()
print("Database disconnected successfully")
