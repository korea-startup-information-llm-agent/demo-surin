# demo-surin

- Meta & Llama Academy Workshop 1기

## 1. 서비스 소개

- 온디바이스 기반 창업 지원 AI: 특허/IP 리스크 관리와 지원사업 가이드를 결합해, 국가 전략산업 창업가의 초기 의사결정을 돕는다.
  - 사용자 예상 질문: https://www.k-startup.go.kr/web/contents/webACTSUPT_LOCAL_FAQ.do

- 타겟층: 국가 10대 전략산업(반도체, 이차전지, 차세대 원자력, 첨단바이오, 우주항공·해양, 수소, 사이버보안, 인공지능, 차세대통신, 첨단로봇·제조, 양자, 첨단모빌리티)에 도전하려는 청년 창업가 및 초기 스타트업 팀

## 2. 데이터셋

- **LLM Fine-tuning** 용: [지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터 (AI Hub)](https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EC%A7%80%EC%8B%9D%EC%9E%AC&aihubDataSe=data&dataSetSn=71843)

- **vectorDB 임베딩** 용: [국가중점기술 대응 특허 데이터 (AI Hub)](https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EC%A7%80%EC%8B%9D%EC%9E%AC&aihubDataSe=data&dataSetSn=71739)
  - 설명: 29,337건의 특허 관련 JSON 데이터 (CC_빅데이터_인공지능: 15,612건, CD_컴퓨팅_소프트웨어: 13,725건)
    - 특허에 대한 기본지식은 가지고 있어야 할 듯함.
  - 가공 후 구성
    
    ```json
    { 
      // Metadata 용
      "register_year": "등록연도",
      "register_month": "등록월",
      "register_day": "등록일",
      "regitster_number": "등록번호",
      
      "application_year": "출원연도",
      "application_month": "출원월",
      "application_day": "출원일",
      "application_number": "출원번호",
      
      "open_year": "공개연도",
      "open_month": "공개월",
      "open_day": "공개일",
      "open_number": "공개번호",
      
      "ipc_section": "IPC-Section",
      "ipc_class": "IPC-Class",
      "ipc_subclass": "IPC-Subclass",
      "ipc_maingroup": "IPC-MainGroup",

      "large_no": "대분류코드",
      "middle_no": "중분류코드",
      "small_no": "소분류코드",

      "country_code": "국가코드",
      "document_id": "문헌키",
      "document_type": "문헌타입",

      // 임베딩용
      "invention_title": "발명의명칭",
      "inventor_name": "발명자명",
      "applicant_name": "출원인명",
      "claims": "청구항",
      "abstract": "요약",
      "keyword": "명칭, 요약, 청구항에서 추출한 키워드 6개 및 국가전략기술 12대 분야 대분류 1개 (리스트 형태로 저장: /로 split하기)"
    }
    ```

## 3. RAG Pipeline

- vectorDB: `pg-vector`
  - 관련 깃허브 링크: https://github.com/pgvector/pgvector-python

- embedding model
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large 
  - upstage embedding model (passage)

- `국가중점기술 대응 특허 데이터`의 raw 데이터의 **명칭**, **요약**, **청구항** 내용을 임베딩하고, 그 외의 raw, label 데이터의 내용을 metadata로 하여 vectorDB를 구성한다.
  - `Langchain_postgres`를 사용하면 pgvector를 Langchain에서 쉽게 사용할 수 있다.

- 완전 Postgresql@17을 install, pgvector를 install 해서 사용한다.  
  *pgvector는 14, 17 버전에서 사용할 수 있다고 한다.
  - postgre에 위의 JSON의 키값을 column name으로 해서 데이터를 넣어준다. 
    - [ ] 함수화해서 자동화할 수 있도록 하기


## 4. LLM Fine-tuning

- 사용 모델: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) or [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

- 파인튜닝 방법: LoRA, 지식재산권 질의응답 데이터를 이용해 파인튜닝하여 특화 모델을 만든다.
  - 이때, 사용자 질문을 함수로 판단하거나 프롬프팅을 이용해 파인튜닝한 모델이 민감정보가 있는지 없는지 판단할 수 있도록 한다.
    - 민감정보가 있다면, 그 부분을 마스킹하여

## 5. Langgraph를 통한 Agent Workflow 만들기

- 우리 서비스 사용 시 주의 사항으로, "법률 자문은 아니지만, 특허 출원 전이라면 가능한 한 원문 공개를 최소화하고, 필요 시 사전에 NDA 또는 사설 배포를 검토하세요." 이러한 내용을 꼭 첨부 해줘야 한다.

### Models
  
  - LLM

    1. Llama-3.2-3B
    2. Solar-pro2 (API)

  - Embedding Model

    1. solar-passage (API)
    2. draguke/bge-m3-ko (BAAI/bge-m3 의 한국어 특화 및 경량화 모델)

### TOOLS
  
  1. retriever_tool: RAG를 위한
  2. select_llm: 어떤 모델을 사용할지 판단
  3. safeguard_tool: 민감정보를 내부 단어로 변경
  4. load_to_table: 행렬을 표로 바꿔주는 
  5. 