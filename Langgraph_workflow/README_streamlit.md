# 🤖 StartMate - 지식재산권 챗봇 (Streamlit 버전)

예비 창업가와 스타트업을 위한 전문적인 지식재산권 안내 챗봇의 Streamlit 웹 인터페이스입니다.

## ✨ 주요 기능

- **💬 대화형 인터페이스**: 직관적인 채팅 UI
- **🔍 지능형 질문 분석**: 질문 유형 자동 분류
- **🛠️ 다양한 도구 활용**: 
  - 질문 상세 분석
  - IP 법령 데이터베이스 검색
  - 특허 데이터베이스 검색
  - 웹 검색
- **📊 실시간 정보 표시**: 도구 사용 정보 및 분석 결과
- **⚙️ 설정 가능**: 로컬/클라우드 모델 선택

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements_streamlit.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# Upstage API (필수)
UPSTAGE_API_KEY=your_upstage_api_key

# Tavily 웹 검색 (필수)
TAVILY_API_KEY=your_tavily_api_key

# Qdrant 벡터 DB (필수)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# 임베딩 모델 (필수)
EMBED_URL=your_embedding_model_url

# 로컬 모델 설정 (선택사항)
USE_LOCAL_LLM=false
LLAMA_MODEL_PATH=./models/qwen2.5-3b-instruct.gguf
LLAMA_N_CTX=2048
LLAMA_N_THREADS=4
```

### 3. 앱 실행

#### 방법 1: 실행 스크립트 사용
```bash
python run_streamlit.py
```

#### 방법 2: 직접 실행
```bash
streamlit run streamlit_app.py
```

### 4. 브라우저에서 접속

앱이 실행되면 브라우저에서 `http://localhost:8501`로 접속하세요.

## 🎯 사용법

### 기본 사용
1. 왼쪽 텍스트 영역에 지식재산권 관련 질문을 입력
2. "📤 질문 전송" 버튼 클릭
3. 챗봇의 답변 확인

### 예시 질문
- "AI 기술을 특허로 보호받고 싶은데 절차가 어떻게 되나요?"
- "상표 등록 비용은 얼마나 드나요?"
- "기존 특허와 유사한지 어떻게 확인하나요?"
- "스타트업에서 지식재산권 전략은 어떻게 세워야 하나요?"

### 사이드바 설정
- **모델 설정**: 로컬/클라우드 모델 선택
- **도구 설정**: 도구 사용 정보 및 분석 정보 표시 여부
- **대화 초기화**: 현재 대화 내용 삭제

## 🏗️ 아키텍처

```
streamlit_app.py (웹 인터페이스)
    ↓
workflow.py (LangGraph 워크플로우)
    ↓
agent_tools.py (도구 및 분석 함수)
    ↓
utils.py (유틸리티 및 상태 정의)
```

### 워크플로우 플로우
```
사용자 질문 → 질문 분류 → 챗봇 → 도구 실행 → 최종 답변
```

## 🔧 고급 설정

### 로컬 모델 사용
1. llama.cpp 설치: `pip install llama-cpp-python`
2. 모델 다운로드 (GGUF 형식)
3. 사이드바에서 "로컬 모델 사용" 체크
4. 모델 경로 및 설정 조정

### 커스터마이징
- `streamlit_app.py`: UI 및 인터페이스 수정
- `workflow.py`: 워크플로우 로직 수정
- `agent_tools.py`: 도구 및 분석 함수 수정

## 🐛 문제 해결

### 일반적인 문제들

1. **패키지 설치 오류**
   ```bash
   pip install --upgrade pip
   pip install -r requirements_streamlit.txt
   ```

2. **환경 변수 오류**
   - `.env` 파일이 올바른 위치에 있는지 확인
   - 환경 변수 이름과 값이 정확한지 확인

3. **모델 연결 오류**
   - API 키가 유효한지 확인
   - 네트워크 연결 상태 확인

4. **벡터 DB 연결 오류**
   - Qdrant 서버 상태 확인
   - 컬렉션 존재 여부 확인

### 로그 확인
앱 실행 시 콘솔에서 상세한 로그를 확인할 수 있습니다.

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

버그 리포트나 기능 제안은 GitHub Issues를 통해 해주세요.

---

**StartMate** - 지식재산권으로 스타트업의 성공을 돕습니다! 🚀
