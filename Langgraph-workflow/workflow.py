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
from langgraph.graph import StateGraph, END
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
os.environ["UPSTAGE_API_KEY"] = settings.UPSTAGE_API_KEY

# 로컬 경량 모델 설정
local_model = ""

# 대형 모델 설정
large_llm = ChatUpstage(model="solar-pro2", temperature=0)

# 도구 목록
TOOLS = [search_ipraw, search_patent, search_in_web]

# llm에 TOOLS 바인딩
llm_with_tools = large_llm.bind_tools(TOOLS)

# 챗봇 함수 정의 - 인천 토박이 친구 페르소나 적용
async def chatbot(state: State):
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
    
    response = await llm_with_tools.ainvoke(messages_with_system)

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

# 그래프 생성 함수
async def make_graph():

    graph_builder = StateGraph(State)

    # 도구 노드 설정
    tool_node = ToolNode(tools=TOOLS)

    # 노드 추가
    graph_builder.add_node("analyze", classify_question_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot", chatbot)

    # 조건부 엣지 추가
    graph_builder.add_conditional_edges(
        "analyze",
        lambda x: "chatbot",  # 분석 후 항상 챗봇으로
        {"chatbot": "chatbot"}
    )

    graph_builder.add_conditional_edges(
        "chatbot",
        # select_next_node,
        {"tools": "tools", "analyze": "analyze", END: END}
    )

    # 엣지 추가하기
    graph_builder.add_edge(START, "analyze")  # 시작 시 질문 분석부터

    graph_builder.add_conditional_edges(
        "tools",
        # after_tools_router,
        {"tools": "tools", "chatbot": "chatbot"}
    )

    # 컴파일
    graph = \
    graph_builder.compile(
        # checkpointer=checkpointer,
    )
    return graph