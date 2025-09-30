# demo-surin

- Meta & Llama Academy Workshop 1기

## 1. 서비스 소개

- 온디바이스 기반 창업 지원 AI: 특허/IP 리스크 관리와 지원사업 가이드를 결합해, 국가 전략산업 창업가의 초기 의사결정을 돕는다.
  - 사용자 예상 질문: https://www.k-startup.go.kr/web/contents/webACTSUPT_LOCAL_FAQ.do

- 타겟층: 국가 10대 전략산업(반도체, 이차전지, 차세대 원자력, 첨단바이오, 우주항공·해양, 수소, 사이버보안, 인공지능, 차세대통신, 첨단로봇·제조, 양자, 첨단모빌리티)에 도전하려는 청년 창업가 및 초기 스타트업 팀

## 2. 데이터셋

- **LLM Fine-tuning** 용: [지식재산권법 LLM 사전학습 및 Instruction Tuning 데이터 (AI Hub)](https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EC%A7%80%EC%8B%9D%EC%9E%AC&aihubDataSe=data&dataSetSn=71843)

- **vectorDB 임베딩** 용: [국가중점기술 대응 특허 데이터 (AI Hub)](https://www.aihub.or.kr/aihubdata/data/view.do?searchKeyword=%EC%A7%80%EC%8B%9D%EC%9E%AC&aihubDataSe=data&dataSetSn=71739)
  - 설명: 29,337건의 JSON 데이터 (CC_빅데이터_인공지능: 15,612건, CD_컴퓨팅_소프트웨어: 13,725건)
  - 구성
    
    ```json

    ```

## 3. RAG Pipeline

- vectorDB: `pg-vector`
  - 관련 깃허브 링크: https://github.com/pgvector/pgvector-python

- embedding model
  - BAAI/bge-m3
  - intfloat/multilingual-e5-large 
  - upstage embedding model (passage)

- `국가중점기술 대응 특허 데이터`의 

## 4. LLM Fine-tuning

- 사용 모델: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) or [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

- 파인튜닝 방법: LoRA