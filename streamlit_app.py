import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# Local imports
from Langgraph_workflow.workflow import make_graph
from Langgraph_workflow.utils import State

# Load environment variables
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="StartMate - 지식재산권 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .tool-info {
        background-color: #fff3e0;
        border-left-color: #ff9800;
        font-size: 0.9rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .analysis-info {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
        font-size: 0.9rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown('<h1 class="main-header">🤖 StartMate - 지식재산권 챗봇</h1>', unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 설정
    st.subheader("모델 설정")
    use_local = st.checkbox("로컬 모델 사용", value=False)
    
    if use_local:
        st.info("로컬 llama.cpp 모델을 사용합니다.")
        model_path = st.text_input("모델 경로", value="./models/qwen2.5-3b-instruct.gguf")
        n_ctx = st.slider("컨텍스트 크기", 512, 4096, 2048)
        n_threads = st.slider("스레드 수", 1, 8, 4)
    else:
        st.info("Upstage Solar 모델을 사용합니다.")
    
    # 도구 설정
    st.subheader("도구 설정")
    show_tool_info = st.checkbox("도구 사용 정보 표시", value=True)
    show_analysis = st.checkbox("질문 분석 정보 표시", value=True)
    
    # 대화 초기화
    if st.button("🗑️ 대화 초기화"):
        st.session_state.clear()
        st.rerun()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph" not in st.session_state:
    st.session_state.graph = make_graph()
if "config" not in st.session_state:
    st.session_state.config = RunnableConfig(
        configurable={"thread_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    )

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 대화")
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <strong>👤 사용자:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'''
            <div class="chat-message assistant-message">
                <strong>🤖 StartMate:</strong><br>
                {message["content"]}
            </div>
            ''', unsafe_allow_html=True)
        
        # 도구 사용 정보 표시
        if show_tool_info and "tool_info" in message:
            st.markdown(f'''
            <div class="tool-info">
                <strong>🔧 사용된 도구:</strong> {message["tool_info"]}
            </div>
            ''', unsafe_allow_html=True)
        
        # 질문 분석 정보 표시
        if show_analysis and "analysis_info" in message:
            st.markdown(f'''
            <div class="analysis-info">
                <strong>🔍 질문 분석:</strong> {message["analysis_info"]}
            </div>
            ''', unsafe_allow_html=True)

with col2:
    st.subheader("📊 시스템 정보")
    
    # 환경 변수 상태
    st.write("**환경 설정:**")
    env_status = {
        "UPSTAGE_API_KEY": "✅ 설정됨" if os.getenv("UPSTAGE_API_KEY") else "❌ 미설정",
        "TAVILY_API_KEY": "✅ 설정됨" if os.getenv("TAVILY_API_KEY") else "❌ 미설정",
        "QDRANT_URL": "✅ 설정됨" if os.getenv("QDRANT_URL") else "❌ 미설정",
        "EMBED_URL": "✅ 설정됨" if os.getenv("EMBED_URL") else "❌ 미설정",
    }
    
    for key, status in env_status.items():
        st.write(f"- {key}: {status}")
    
    # 사용 가능한 도구
    st.write("**사용 가능한 도구:**")
    tools = [
        "🔍 analyze_question - 질문 상세 분석",
        "📚 search_ipraw - IP 법령 검색",
        "📄 search_patent - 특허 검색",
        "🌐 search_in_web - 웹 검색"
    ]
    for tool in tools:
        st.write(f"- {tool}")

# 사용자 입력
st.subheader("💭 질문하기")
user_input = st.text_area(
    "지식재산권 관련 질문을 입력하세요:",
    placeholder="예: AI 기술을 특허로 보호받고 싶은데 절차가 어떻게 되나요?",
    height=100
)

col_send, col_clear = st.columns([3, 1])

with col_send:
    send_button = st.button("📤 질문 전송", type="primary")

with col_clear:
    if st.button("🗑️ 입력 초기화"):
        st.rerun()

# 질문 처리
if send_button and user_input.strip():
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # 로딩 표시
    with st.spinner("🤖 답변을 생성하고 있습니다..."):
        try:
            # 워크플로우 실행
            human_message = HumanMessage(content=user_input)
            
            # 도구 사용 정보와 분석 정보를 저장할 변수
            tool_info = []
            analysis_info = {}
            
            # 그래프 실행 및 이벤트 처리
            for event in st.session_state.graph.stream(
                {"messages": [human_message]},
                config=st.session_state.config,
                stream_mode="updates"
            ):
                # 도구 사용 정보 수집
                if "tools" in event and show_tool_info:
                    tool_messages = event["tools"]["messages"]
                    for tool_message in tool_messages:
                        if hasattr(tool_message, "tool_calls") and tool_message.tool_calls:
                            for tool_call in tool_message.tool_calls:
                                tool_info.append(tool_call["name"])
                
                # 질문 분석 정보 수집
                if "classify_question" in event and show_analysis:
                    question_analysis = event["classify_question"].get("question_analysis", {})
                    analysis_info = {
                        "질문 유형": question_analysis.get("question_type", "unknown"),
                        "IP 주제": question_analysis.get("ip_topic", "미분류"),
                        "의도": question_analysis.get("question_intent", "미분류")
                    }
                
                # 챗봇 응답 수집
                if "chatbot" in event:
                    chatbot_response = event["chatbot"]["messages"][-1]
                    if hasattr(chatbot_response, "content"):
                        assistant_content = chatbot_response.content
            
            # 어시스턴트 메시지 추가
            assistant_message = {
                "role": "assistant",
                "content": assistant_content
            }
            
            # 도구 정보 추가
            if tool_info:
                assistant_message["tool_info"] = ", ".join(tool_info)
            
            # 분석 정보 추가
            if analysis_info:
                assistant_message["analysis_info"] = f"유형: {analysis_info['질문 유형']}, 주제: {analysis_info['IP 주제']}, 의도: {analysis_info['의도']}"
            
            st.session_state.messages.append(assistant_message)
            
        except Exception as e:
            error_message = {
                "role": "assistant",
                "content": f"죄송합니다. 오류가 발생했습니다: {str(e)}"
            }
            st.session_state.messages.append(error_message)
    
    # 페이지 새로고침
    st.rerun()

# 예시 질문들
st.subheader("💡 예시 질문")
example_questions = [
    "AI 기술을 특허로 보호받고 싶은데 절차가 어떻게 되나요?",
    "상표 등록 비용은 얼마나 드나요?",
    "기존 특허와 유사한지 어떻게 확인하나요?",
    "스타트업에서 지식재산권 전략은 어떻게 세워야 하나요?",
    "소프트웨어 저작권 보호 범위는 어디까지인가요?"
]

for i, question in enumerate(example_questions):
    if st.button(f"📝 {question}", key=f"example_{i}"):
        st.session_state.example_question = question
        st.rerun()

# 예시 질문이 선택된 경우
if "example_question" in st.session_state:
    st.text_area(
        "선택된 예시 질문:",
        value=st.session_state.example_question,
        height=50,
        disabled=True
    )
    if st.button("이 질문으로 전송"):
        user_input = st.session_state.example_question
        del st.session_state.example_question
        st.rerun()

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 StartMate - 지식재산권 챗봇 | Powered by LangGraph & Streamlit</p>
        <p>예비 창업가와 스타트업을 위한 전문적인 지식재산권 안내 서비스</p>
    </div>
    """,
    unsafe_allow_html=True
)
