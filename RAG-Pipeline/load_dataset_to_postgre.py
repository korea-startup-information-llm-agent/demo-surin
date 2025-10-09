# Postgres에 데이터 넣기
# 데이터 로더 만들기

import json
import ast
import psycopg2

db = psycopg2.connect(host="localhost", database="postgres", user="ssr", port=5432)
print("Database connected successfully")

# 데이터 조작을 위한 객체 생성
cursor = db.cursor()

# json 파일 읽기
with open("RAG-Pipeline/dataset/for_embedding.json", "r", encoding="utf-8") as f:
    total_data = json.load(f)

print(f"Total data: {len(total_data)}")

for data in total_data:
    
    # 필요한 필드 추출하기
    document_id = data["document_id"]
    document_type = data["document_type"]
    country_code = data["country_code"]
    application_number = data["application_number"]
    application_year = data["application_year"]
    application_month = data["application_month"]
    application_day = data["application_day"]

    # 출원일 생성
    if application_year and application_month and application_day:
        application_date = f"{application_year}-{application_month}-{application_day}"
    else:
        application_date = None

    ipc_all = data["ipc_all"]
    invention_title = data["invention_title"]
    inventor_name = data["inventor_name"]
    applicant_name = data["applicant_name"]
    claims = data["claims"]
    abstract = data["abstract"]

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
    
    # 이미 들어있는지 확인
    cursor.execute("SELECT EXISTS (SELECT 1 FROM patents WHERE document_id=%s)", (document_id,))
    if cursor.fetchone()[0]:
        print(f"Patents with document_id {document_id} already exists in the database.")
        continue
    
    # SQL query 작성
    query = (
        "INSERT INTO patents (document_id, document_type, country_code, application_number, application_date, ipc_all, invention_title, applicant_name, claims, abstract, keyword) "
        "VALUES (%s, %s, %s, %s, TO_DATE(NULLIF(%s, ''), 'YYYY-MM-DD'), %s, %s, %s, %s, %s, %s)"
    )
    input_data = (document_id, document_type, country_code, application_number, application_date, ipc_all, invention_title, applicant_name, claims, abstract, keyword)
    # 실행
    cursor.execute(query, input_data)
    db.commit()
    print(f"Patents with document_id {document_id} inserted into the database.")

cursor.close()
db.close()


