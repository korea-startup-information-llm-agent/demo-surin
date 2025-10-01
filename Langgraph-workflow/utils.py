import requests

# 허깅페이스 스페이스에서 임베딩 모델 불러오기
class HFSpaceEmbeddingFunction:
    def __init__(self, space_url: str):
        self.space_url = space_url  # e.g., "https://your-username-your-space-name.hf.space"

    def __call__(self, text: str) -> list:
        response = requests.post(
            self.space_url,
            json={"inputs": text}  # 이 구조는 Space 구현에 따라 달라질 수 있음
        )
        response.raise_for_status()
        return response.json()["embedding"]  # 응답 형식에 따라 키가 다를 수 있음


# 상태 정의하기
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # LLM과의 대화 메시지
    question_analysis: dict                  # LLM이 분석한 질문 정보



# # 조건부 논리 정의 함수
# def select_next_node(state: State):

#     # 최근 어떤 ToolMessage가 붙었는지
#     dump_tool_names(state.get("messages", []))
    
#     # 질문 분석이 되지 않았다면
#     if "question_analysis" not in state or not state["question_analysis"]:
#         return "analyze"

#     qa_types = state["question_analysis"].get("question_types", {})

#     # 디버깅
#     # print(f"[DEBUG] select_next_node - qa_types: {qa_types}")
#     logger.info(f"[DEBUG] select_next_node - qa_types: {qa_types}")

#     # 길찾기 인텐트(route=True)면 바로 tools 실행
#     if qa_types.get("route"):
#         if called_tool(state, "build_kakaomap_route"):
#             return END
#         return "tools"


#     # 나머지 일반 도구 선택하기
#     return tools_condition(state)

# def after_tools_router(state):
#     return "tools" if has_unresolved_tool_calls(state.get("messages", [])) else "chatbot"
