# langgraph workflow

# =============================================================================
# ALL IMPORTS - ORGANIZED BY CATEGORY
# =============================================================================

# Standard Library
import os
import json
import csv
from datetime import datetime
from collections import Counter

# Third-party Libraries
import numpy as np
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI

# LangChain Core
from langchain_core.documents import Document
from langchain_core.messages import (
    BaseMessage, 
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage
)
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

# LangChain Upstage
from langchain_upstage import ChatUpstage

# LangGraph
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# Pydantic
from pydantic import BaseModel

# Typing
from typing import (
    List, 
    Dict, 
    Any, 
    TypedDict, 
    Annotated, 
    Sequence, 
    Literal
)

from agent_tools import *
from utils import *

# Load environment variables
load_dotenv(verbose=True)

print("✅ All imports successful!")

# 환경 변수 설정
os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")

# 로컬 경량 모델 설정
local_model = ""

# 대형 모델 설정
large_llm = ChatUpstage(model="solar-pro2", temperature=0)

# 도구 목록
TOOLS = [analyze_question, search_ipraw, search_patent, search_in_web]

# llm에 TOOLS 바인딩
llm_with_tools = large_llm.bind_tools(TOOLS)

# 챗봇 함수 정의 - 인천 토박이 친구 페르소나 적용
def chatbot(state: State):
    # 질문 분석 결과 확인
    question_analysis = state.get("question_analysis", {})

    # 도구 결과 확인
    analyze_question_results = state.get("analyze_question_results", [])
    ipraw_results = state.get("ipraw_results", [])
    patent_results = state.get("patent_results", [])
    search_in_web_results = state.get("search_in_web_results", [])

    # 시스템 메시지에 페르소나 설정
    system_message = SystemMessage(
        content=f"""
        당신은 예비 창업가와 스타트업을 위한 전문적인 지식재산권 안내 챗봇입니다.

        당신의 주요 역할은 다음과 같습니다:

        1. 사용자의 질문을 이해하고, 관련된 지식재산권 범주(예: 특허, 상표, 디자인, 저작권 등)로 분류합니다.
        2. 질문의 의도(예: 출원 절차, 등록 가능성, 비용, 침해 여부 등)를 파악하여 정확하고 친절하게 설명합니다.
        3. 법령, 제도, 절차 등을 바탕으로 구체적이고 신뢰할 수 있는 정보를 제공합니다.
        4. 전문 용어는 알기 쉽게 풀어 설명하며, 사용자가 이해하기 쉬운 언어로 응답합니다.
        5. 단순한 요약이나 정의를 넘어, 사용자의 실제 상황에 맞는 방향을 제안합니다.
        6. 사용자의 질문이 모호하거나 불완전할 경우, 명확히 하기 위한 추가 질문을 합니다.
        7. 법적 판단, 권리 침해 판별, 구체적인 출원 전략은 제공하지 않으며, 그럴 경우 전문가 상담을 권장합니다.

        지켜야 할 커뮤니케이션 원칙:

        - 항상 존중하고 공감하는 태도로 응답합니다.
        - 중립적인 관점에서 설명하며, 확정적인 법적 조언은 피합니다.
        - 질문이 너무 복잡하거나 민감한 경우, “이 부분은 전문가와 상의하시길 권장드립니다”라고 안내합니다.
        - 사용자 질문이 너무 짧거나 모호한 경우, 추가 설명을 요청하거나 보충 질문을 합니다.

        당신은 지식재산권 도우미로서, 특허청이나 공공기관에서 제공하는 정보를 기반으로 설명할 수 있으며, 필요 시 검색이나 외부 도구를 활용하여 정보를 제공합니다.

        """
    )

    # if has_unresolved_tool_calls(state["messages"]):
    #     return {}    # 상태 변경 없이 다음 노드로
    
    # 시스템 메시지 추가
    messages_with_system = [system_message] + state["messages"]
    
    response = llm_with_tools.invoke(messages_with_system)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"도구 호출: {tool_call['name']}")
    # # 디버깅
    # # print(f"[DEBUG] LLM 응답: {response}")
    # logger.info(f"[DEBUG] LLM 응답: {response}")
    # if hasattr(response, 'tool_calls') and response.tool_calls:
    #     # print(f"[DEBUG] 도구 호출 감지: {response.tool_calls}")
    #     logger.info(f"[DEBUG] 도구 호출 감지: {response.tool_calls}")

    # # [Fallback] tool_calls가 없고, 답변이 필러 멘트이면 -> 강제 tool call 1회
    # if (not getattr(response, "tool_calls", None)) and is_filter(getattr(response, "content", "")):
    #     forced = make_forced_tool_ai_message(state)
    #     if forced is not None:
    #         # 여기서 바로 ToolNode가 실행될 수 있도록 AIMessage(tool_calls=_)를 반환
    #         return {"messages": [forced]}

    # 메시지 호출 및 반환
    return {"messages": [response]}


def tool_call_or_end(state: State):

    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return "end"
    else:
        return "tool_call_needed"


tools_by_name = {tool.name: tool for tool in TOOLS}

def tool_node(state: State) -> Dict[str, Any]:
    """Execute tools and update state with structured results."""
    
    outputs = []
    update = {"messages": []}

    # 마지막 메시지에 tool_calls가 있다면 이름과 args를 가져와서 결과를 반환한다.
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = None
        for tool in TOOLS:
            if tool.name == tool_call["name"]:
                tool_result = tool.invoke(tool_call["args"])
                break
        
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result) if tool_result else "Tool not found",
                tool_call_id=tool_call["id"],
            )
        )

        if tool_call["name"] == "analyze_question":
            analyze_question_results = tool_result if isinstance(tool_result, list) else []
            update["analyze_question_results"] = analyze_question_results
            print(f"📚 analyze_question completed with {len(analyze_question_results)} results")

        if tool_call["name"] == "search_ipraw": # tool_call이 search_ipraw인 경우
            ipraw_results = tool_result if isinstance(tool_result, list) else []
            update["ipraw_results"] = ipraw_results # ipraw_results를 업데이트 해줍니다.
            print(f"📚 search_ipraw completed with {len(ipraw_results)} results")
        
        if tool_call["name"] == "search_patent": # tool_call이 search_patent인 경우
            patent_results = tool_result if isinstance(tool_result, list) else []
            update["patent_results"] = patent_results # patent_results를 업데이트 해줍니다.
            print(f"📚 search_patent completed with {len(patent_results)} results")

        if tool_call["name"] == "search_in_web": # tool_call이 web_search인 경우
            web_results = tool_result if isinstance(tool_result, list) else []
            update["search_in_web_results"] = web_results # web_search_raw_results를 업데이트 해줍니다.
            print(f"🌐 Web search completed with {len(web_results)} results")
    
    update["messages"] = outputs # 위에서 정의한 outputs를 업데이트 해줍니다.
    return update


# 그래프 생성 함수
def make_graph():

    graph_builder = StateGraph(State)

    # 노드 추가
    graph_builder.add_node("classify_question", classify_question_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)

    # 엣지 추가하기
    graph_builder.add_edge(START, "classify_question")  # 시작 시 질문 분석부터
    
    # 조건부 엣지 추가
    graph_builder.add_edge(
        "classify_question", "chatbot",  # 분석 후 항상 챗봇으로
    )

    graph_builder.add_conditional_edges(
        "chatbot",
        tool_call_or_end,
        {"tool_call_needed": "tools", "end": END}
    )
    
    memory = InMemorySaver()

    # 컴파일
    graph = \
    graph_builder.compile(
        checkpointer=memory,
    )
    return graph



if __name__ == "__main__":
    graph = make_graph()
    print("그래프 컴파일 완료")

    

    config = RunnableConfig(
        configurable= {
            "thread_id": "test"
        }
    )

    print("="*100)
    print("StartMate에 오신 것을 환영합니다!")

    
    while True:
        user_input = input("질문: ")

        if user_input.lower() in ["종료", "quit", "exit", "bye"]:
            print("StartMate를 종료합니다.")
            break

        # 빈 입력 처리하기
        if not user_input.strip():
            print("질문을 입력해주세요.")
            continue

        print("답변을 생성중입니다... 🤖\n")

        # 사용자의 메시지를 HumanMessage로 변환
        human_message = HumanMessage(content=user_input)

        # 그래프 실행
        try:
            for event in graph.stream(
                {"messages": [human_message]},
                config=config,
                stream_mode="updates"
            ):
                if "tools" in event:
                    tool_messages = event["tools"]["messages"]
                    for tool_message in tool_messages:
                        if hasattr(tool_message, "tool_calls") and tool_message.tool_calls:
                            for tool_call in tool_message.tool_calls:
                                print(f"도구 호출: {tool_call['name']}")
                                print(f"도구 인수: {tool_call['args']}")

                        elif hasattr(tool_message, "content"):
                            print(f"도구 응답: {tool_message.content[:100]}")

                if "chatbot" in event:
                    chatbot_response = event["chatbot"]["messages"][-1]
                    if hasattr(chatbot_response, "content"):
                        print("답변: ",chatbot_response.content)
                        print("-" * 50)

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            continue
            
    