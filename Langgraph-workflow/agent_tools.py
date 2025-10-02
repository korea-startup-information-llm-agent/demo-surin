import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# 웹 검색 관련
from tavily import TavilyClient
from openai import OpenAI

# Qdrant 관련
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Huggingface embedding 모델 관련
from utils import *

# tool annotation
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# llama.cpp 관련
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# tavily_client 설정
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient()

# 클라이언트 설정
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

# 임베딩 모델 huggingface space로 불러와서 쓰기
EMBED_URL = os.getenv("EMBED_URL") + "/embed"

embedding_function = HFSpaceEmbeddingFunction(EMBED_URL)

# 벡터 DB 불러오기
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 지식재산권법 관련 retriever
ipraw_db = Qdrant(
    client=qdrant_client,
    collection_name="ipraw_db",
    embedding_function=embedding_function
)

ipraw_retriever = ipraw_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.1,
        "k": 5,
    }
)

# 특허 관련 retriever
patent_db = Qdrant(
    client=qdrant_client,
    collection_name="patent_db",
    embedding_function=embedding_function
)

patent_retriever = patent_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.1,
        "k": 5
    }
)

# vectordb search tool
@tool("ipraw_db_search")
def search_ipraw(query: str) -> list:
    """Use this tool to search information about IP raws from the IP raw vector database."""
    docs = ipraw_retriever.get_relevant_documents(query)
    # 메타 데이터로 답변 거르기 과정 필요
    return [
        {
            "content": doc.page_content,
            "metadata": getattr(doc, "metadata", {}),
        }
        for doc in docs
    ]

@tool("patent_db_search")
def search_patent(query: str) -> list:
    """Use this tool to search information about patents from the patent vector database."""
    docs = patent_retriever.get_relevant_documents(query)
    # 메타 데이터로 답변 거르기 과정 필요
    return [
        {
            "content": doc.page_content,
            "metadata": getattr(doc, "metadata", {}),
        }
        for doc in docs
    ]


# 헬퍼 함수
def rewrite_query_for_search(query: str) -> str:
    """
    Use LLM to rewrite customer question into optimal search query.
    
    Args:
        query: Original search query
        
    Returns:
        LLM-optimized search query
    """
    try:
        # 지금 시간을 반영하기 위한 방법
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        
        rewrite_prompt = f"""
        <role>
        You are an expert search query optimizer. Your task is to transform verbose, conversational user questions into concise, keyword-driven search queries that will yield the best possible results from a web search engine.
        </role>

        <instructions>
        1.  **Identify Core Intent:** Analyze the user's question to understand their fundamental goal.
        2.  **Extract Key Entities:** Pull out essential keywords, names, locations, constraints (like budget or time), and concepts.
        3.  **Remove Filler:** Discard conversational fluff (e.g., "I was wondering", "can you help me", "please", "I think").
        4.  **Synthesize Keywords:** Combine the extracted entities into a logical, concise search string.
        5.  **Keep it Brief:** The final query should ideally be under 10 words.
        6.  **IMPORTANT:** Return ONLY the search query, no explanations, no tags, no thinking process.
        7.  **Time Context:** If user asks time-specific question, use current time information (current date: {current_date}).
        </instructions>

        <example>
        <user_question>
        I'm thinking of going to Europe this winter, maybe for like a week. I'm on a budget but I still want to see some cool historical stuff. Can you give me some recommendations?
        </user_question>
        <rewritten_query>
        best budget winter destinations Europe historical sites one week
        </rewritten_query>
        </example>

        <user_question>
        {query}
        </user_question>

        <output_format>
        Provide ONLY the rewritten search query as a single line of text. Do not add any introductory phrases like "Here is the query:".
        And also DO NOT PROVIDE THE THINKING STEPS, just provide the rewritten query.
        </output_format>

        <rewritten_query>
        """
        
        # 구조화된 출력을 뽑을 수 있도록!
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "search_query_suggestions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "rewritten_query": {
                            "type": "string",
                            "description": "The most relevant search query for this inquiry"
                        }
                    },
                    "required": ["rewritten_query"]
                }
            }
        }
    
        response = client.chat.completions.create(
            model="solar-pro2",
            messages=[{"role": "system", "content": rewrite_prompt}],
            # max_tokens=100,
            response_format=response_format
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Return the primary query as the main result
        return result["rewritten_query"]
        
    except Exception as e:
        print(f"   ⚠️ LLM query rewrite failed: {str(e)}, using original query")
        return query

# Web Search
@tool("search_in_web")
def search_in_web(query: str, rewrite_mode: bool = True) -> list:
    """Search the web for relevant information about a customer query.
    
    This tool searches the internet to find current information that can help
    answer customer questions, especially for technical issues or general topics
    not covered in the internal knowledge base.
    
    Args:
        query: The customer's question or search query
        max_results: Maximum number of search results to return (default: 3)
        rewrite_mode: Whether to use LLM to optimize the search query (default: True)
        
    Returns:
        List of dictionaries containing structured web search results with scores and metadata
    """
    try:
        search_query = query
        if rewrite_mode:
            search_query = rewrite_query_for_search(query)

        response = tavily_client.search(
            query=search_query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_images=True,
        )

        results = []

        for result in response.get("results", []):
            score = result.get("score")
            if score and score > 0.5 :
                results.append({
                    "title": result.get("title", ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': score
                })

        return results

    except Exception as e:
        return [{"error": f"Web search error: {str(e)}"}]


# 사용자 질문 분기 처리하기
def _get_analysis_client_and_model():
    """
    Return (client, model) for analysis. Supports llama.cpp local models.
    Env:
      - USE_LOCAL_LLM: "1" or "true" to prefer local llama.cpp
      - LLAMA_MODEL_PATH: path to GGUF model file (e.g., "./models/qwen2.5-3b-instruct.gguf")
      - LLAMA_N_CTX: context window size (default: 2048)
      - LLAMA_N_THREADS: number of threads (default: 4)
      - UPSTAGE_API_KEY: used when local not enabled
    """
    global client

    use_local = os.getenv("USE_LOCAL_LLM", "").lower() in ("1", "true", "yes")
    
    if use_local and LLAMA_CPP_AVAILABLE:
        model_path = os.getenv("LLAMA_MODEL_PATH", "").strip()
        if not model_path:
            # 기본 모델 경로들 시도
            default_paths = [
                "./models/qwen2.5-3b-instruct.gguf",
                "./models/llama-3.2-3b-instruct.gguf", 
                "./models/phi-3-mini-4k-instruct.gguf",
                "./llama.cpp/models/qwen2.5-3b-instruct.gguf"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print("⚠️ 로컬 모델 파일을 찾을 수 없습니다. 클라우드 모델을 사용합니다.")
                use_local = False
    
    if use_local and LLAMA_CPP_AVAILABLE and model_path:
        try:
            # llama.cpp 클라이언트 생성
            llama_client = Llama(
                model_path=model_path,
                n_ctx=int(os.getenv("LLAMA_N_CTX", "2048")),
                n_threads=int(os.getenv("LLAMA_N_THREADS", "4")),
                verbose=False
            )
            
            # OpenAI 호환 래퍼 클래스
            class LlamaOpenAIWrapper:
                def __init__(self, llama_client):
                    self.llama_client = llama_client
                
                class ChatCompletion:
                    def __init__(self, llama_client):
                        self.llama_client = llama_client
                    
                    def create(self, model=None, messages=None, temperature=0.1, response_format=None, **kwargs):
                        # 시스템 메시지와 사용자 메시지 분리
                        system_msg = ""
                        user_msg = ""
                        
                        for msg in messages:
                            if msg["role"] == "system":
                                system_msg = msg["content"]
                            elif msg["role"] == "user":
                                user_msg = msg["content"]
                        
                        # 프롬프트 구성
                        if system_msg:
                            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
                        else:
                            prompt = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
                        
                        # JSON 스키마가 있으면 추가 지시사항 포함
                        if response_format and "json_schema" in response_format:
                            prompt += "응답은 반드시 지정된 JSON 스키마 형식으로만 제공하세요. 다른 설명은 포함하지 마세요.\n"
                        
                        # 모델 실행
                        response = self.llama_client(
                            prompt,
                            max_tokens=512,
                            temperature=temperature,
                            stop=["<|im_end|>", "\n\n"],
                            echo=False
                        )
                        
                        content = response["choices"][0]["text"].strip()
                        
                        # JSON 스키마 응답인 경우 파싱 시도
                        if response_format and "json_schema" in response_format:
                            try:
                                import json
                                # JSON 부분만 추출
                                if "{" in content and "}" in content:
                                    start = content.find("{")
                                    end = content.rfind("}") + 1
                                    json_str = content[start:end]
                                    json.loads(json_str)  # 유효성 검사
                                    content = json_str
                            except:
                                pass
                        
                        return type('Response', (), {
                            'choices': [type('Choice', (), {
                                'message': type('Message', (), {'content': content})()
                            })()]
                        })()
                
                def __init__(self, llama_client):
                    self.chat = self.ChatCompletion(llama_client)
            
            wrapper_client = LlamaOpenAIWrapper(llama_client)
            return wrapper_client, "llama-cpp-local"
            
        except Exception as e:
            print(f"⚠️ 로컬 llama.cpp 모델 로딩 실패: {str(e)}, 클라우드 모델을 사용합니다.")
            use_local = False
    
    # 기본값: Upstage 클라우드 모델 사용
    if not use_local:
        model = os.getenv("ANALYSIS_MODEL", "solar-pro2")
        return client, model



# 질문 분석 tool
@tool("analyze_question")
def analyze_question(user_question: str) -> dict:
    """사용자의 질문을 분석해서 필요한 정보를 추출합니다."""
    
    extracted_info = {
        "ip_topic": None,                  # 질문 주제 (e.g., "특허 출원", "상표 등록", "디자인권", "저작권", "IP 전략")
        "is_about_patent": False,         # 특허 관련 여부
        "is_about_trademark": False,      # 상표 관련 여부
        "is_about_copyright": False,      # 저작권 관련 여부
        "is_about_design": False,         # 디자인권 관련 여부

        "question_intent": None,          # 의도 (e.g., "출원 절차 문의", "출원 가능성 판단", "비용 문의", "권리 보호 범위", "침해 여부", "기존 특허 조사")

        "needs_professional_help": False, # 전문가 상담이 필요한 복잡한 질문인지
        "jurisdiction": None,             # 국가/관할 구역 (e.g., "대한민국", "미국", "PCT", "유럽")

        "technology_field": None,         # 기술 분야 (e.g., "AI", "헬스케어", "핀테크", "화장품")
        "has_existing_product": False,    # 기존 제품이나 아이디어가 있는지 여부
        "product_description": None,      # 아이디어/제품 설명 요약
        "is_confidential": None,          # 기밀 정보 포함 여부 (보안 필요 시 표시)

        "needs_clarification": False,     # 질문이 모호하거나 보완 설명 필요한지
        "clarification_question": None    # 명확히 하기 위한 질문
    }

    text = user_question.lower()

    # 1) 룰 베이스 기반으로 나누기
    topic_keywords = {
        "patent": ["특허", "출원", "명세서", "선행기술", "pct", "우선권 주장", "발명"],
        "trademark": ["상표", "브랜드", "로고", "서비스표", "출원번호", "등록번호"],
        "copyright": ["저작권", "저작물", "저작자의", "라이선스", "license", "저작"],
        "design": ["디자인권", "의장", "도안", "디자인 등록"],
    }
    intent_map = {
        "procedure": ["절차", "방법", "어떻게", "과정", "flow"],
        "feasibility": ["가능성", "될까요", "될까", "요건", "충족"],
        "cost": ["비용", "수수료", "견적", "가격"],
        "scope": ["범위", "권리범위", "해석", "청구항"],
        "infringement": ["침해", "분쟁", "소송", "경고장"],
        "search": ["조사", "검색", "선행기술", "prior art", "조회"],
    }
    jurisdiction_map = {
        "대한민국": ["한국", "대한민국", "kipo", "특허청"],
        "해외": ["해외", "외국"],
        # "미국": ["미국", "us", "uspto"],
        # "유럽": ["유럽", "euipo", "epo", "유럽특허"],
        # "pct": ["pct", "국제출원"],
        # "중국": ["중국", "cnipa"],
        # "일본": ["일본", "jpo"],
    }
    tech_map = {
        "ict-sw": ["빅데이터•인공지능", "컴퓨팅•소프트웨어"],
    }

    # Topic flags
    if any(topic_keyword in text for topic_keyword in topic_keywords["patent"]):
        extracted_info["is_about_patent"] = True
        if not extracted_info["ip_topic"]:
            extracted_info["ip_topic"] = "특허"

    if any(topic_keyword in text for topic_keyword in topic_keywords["trademark"]):
        extracted_info["is_about_trademark"] = True
        if not extracted_info["ip_topic"]:
            extracted_info["ip_topic"] = "상표"

    if any(topic_keyword in text for topic_keyword in topic_keywords["copyright"]):
        extracted_info["is_about_copyright"] = True
        if not extracted_info["ip_topic"]:
            extracted_info["ip_topic"] = "저작권"

    if any(topic_keyword in text for topic_keyword in topic_keywords["design"]):
        extracted_info["is_about_design"] = True
        if not extracted_info["ip_topic"]:
            extracted_info["ip_topic"] = "디자인권"

    # Intent
    for intent, words in intent_map.items():
        if any(word in text for word in words):
            extracted_info["question_intent"] = {
                "procedure": "출원 절차 문의",
                "feasibility": "출원 가능성 판단",
                "cost": "비용 문의",
                "scope": "권리 보호 범위",
                "infringement": "침해 여부",
                "search": "기존 특허 조사",
            }[intent]
            break

    # Jurisdiction
    for name, words in jurisdiction_map.items():
        if any(word in text for word in words):
            extracted_info["jurisdiction"] = name
            break

    # Technology field
    for name, words in tech_map.items():
        if any(word in text for word in words):
            extracted_info["technology_field"] = name
            break

    # Existing product and description heuristics
    has_product_phrases = ["제품", "서비스", "앱", "장치", "디바이스", "아이디어", "프로토타입", "기능"]
    extracted_info["has_existing_product"] = any(word in text for word in has_product_phrases)

    # crude short description snippet
    if extracted_info["has_existing_product"]:
        # pick a short window as a naive description
        extracted_info["product_description"] = user_question.strip()[:200]

    # Confidentiality signals
    if any(word in text for word in ["비밀", "내부", "공개하지", "nda", "기밀"]):
        extracted_info["is_confidential"] = True

    # Clarification need
    too_short = len(user_question.strip()) < 15
    no_topic = not any([extracted_info["is_about_patent"], extracted_info["is_about_trademark"],
                        extracted_info["is_about_copyright"], extracted_info["is_about_design"]])

    no_intent = extracted_info["question_intent"] is None

    extracted_info["needs_clarification"] = too_short or (no_topic and no_intent)

    if extracted_info["needs_clarification"]:
        extracted_info["clarification_question"] = "질문을 조금 더 구체적으로 설명해 주실 수 있을까요? (주제, 국가, 목적, 제품/아이디어 여부 등)"

    
    return extracted_info




# 사용자 질문 분기 처리하기
def classify_question_node(state: dict) -> dict:
    """
    사용자 질문의 기본 유형을 분류하는 노드
    """
    # 마지막 사용자 메시지 찾기
    user_messages = [msg for msg in state["messages"] if hasattr(msg, 'content') and msg.__class__.__name__ == 'HumanMessage']
    if not user_messages:
        return state
    
    user_message = user_messages[-1].content
    
    # 질문 유형 분류
    prompt = f"""
    You are a startup knowledge assistant that classifies user questions into one of several types.

    Choose ONE of the following `question_type` values based on the user input:
    - summary: 요약이나 개요를 요청하는 질문
    - inmemory: 기억된 정보나 학습된 지식을 요청하는 질문
    - outmemory: 외부 검색이나 최신 정보가 필요한 질문
    - clarification: 명확화나 추가 설명이 필요한 질문
    - generation: 새로운 내용 생성이나 창작을 요청하는 질문
    - intent_classification: 의도나 목적을 파악하려는 질문

    Only return the type as a single lowercase word with no explanation.

    [USER]
    {user_message}
    [/USER]
    [TAG]
    """

    try:
        ai_client, model = _get_analysis_client_and_model()
        response = ai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        question_type = response.choices[0].message.content.strip().lower()
        
    except Exception as e:
        print(f"⚠️ 질문 분류 실패: {str(e)}, 기본값 사용")
        question_type = "clarification"

    # question_analysis에 기본 분류 결과만 저장
    current_analysis = state.get("question_analysis", {})
    updated_analysis = {
        **current_analysis,
        "question_type": question_type
    }

    # 분류 결과를 시스템 메시지로 추가해서 LLM이 바로 활용할 수 있도록 한다.
    classification_message = SystemMessage(
        content=f"""
        [질문 분류 결과]
        질문 유형: {question_type}

        이 분류 결과를 바탕으로 적절한 TOOL을 사용하여 사용자에게 정확한 답변을 제공하세요.

        사용 가능한 도구:
        - analyze_question: 질문을 상세 분석하여 IP 관련 정보를 추출합니다.
        - search_ipraw: 지식재산권법 데이터베이스에서 상표, 저작권, 디자인 관련 정보를 검색합니다.
        - search_patent: 특허 데이터베이스에서 관련 특허 정보를 검색합니다.
        - search_in_web: 웹에서 최신 정보와 일반적인 지식재산권 정보를 검색합니다.
        
        질문 유형에 따라 적절한 도구를 사용하세요:
        - outmemory: 웹 검색이나 데이터베이스 검색 도구 사용
        - clarification: 명확화 질문 생성
        - summary: 기존 지식으로 요약 제공
        - generation: 창의적인 답변 생성
        """
    )

    return {
        **state,
        "question_analysis": updated_analysis,
        "messages": state["messages"] + [classification_message]
    }





    