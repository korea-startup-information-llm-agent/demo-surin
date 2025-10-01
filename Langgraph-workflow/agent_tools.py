# 웹 검색 관련
from tavily import TavilyClient

# Qdrant 관련
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Huggingface embedding 모델 관련
from utils import *

# tavily_client 설정
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient()

# 클라이언트 설정
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

# 임베딩 모델 huggingface space로 불러와서 쓰기
embedding_function = HFSpaceEmbeddingFunction("https://your-space-url.hf.space")

# 벡터 DB 불러오기
qdrant_client = QdrantClient(
    host="localhost",    # 변경 예정
    port=6333
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
        "score_threshold": 0.5,
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
        "score_threshold": 0.5,
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
            query=query,
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
def classify_question_node(state: dict) -> dict:
    user_message = state["messages"][-1]["content"]
    
    prompt = f"""
    You are a startup knowledge assistant that classifies user questions into one of several types.

    Choose ONE of the following `question_type` values based on the user input:
    - summary
    - inmemory
    - outmemory
    - clarification
    - generation
    - intent_classification

    Only return the type as a single lowercase word with no explanation.

    [USER]
    {user_message}
    [/USER]
    [TAG]
    """

    response = client.chat.completions.create(
        model="solar-pro2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    question_type = response.choices[0].message.content.strip().lower()

    return {
        **state,
        "question_type": question_type
    }

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