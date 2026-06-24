import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# 에러가 나던 chains 모듈 대신, 절대 고장 나지 않는 core 모듈(LCEL)을 사용합니다!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("🤖 순수 LCEL 기반의 초고속 Gemini 챗봇을 준비 중입니다...")

# 1. API 키 설정
os.environ["GOOGLE_API_KEY"] = "AIzaSyCq-S8-bGkYabUIzR_FG9yS_nB6kZkB10s"

# 2. 뇌(벡터 DB) 불러오기
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. 입(Gemini LLM) 불러오기
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 4. 프롬프트 세팅
template = """당신은 공공기관의 IT/보안 규정을 안내하는 전문 AI 어시스턴트입니다.
반드시 아래에 제공된 문서 내용(Context)을 바탕으로 질문에 답변하세요.
문서에 없는 내용이라면 지어내지 말고 '제공된 가이드라인에서는 해당 내용을 찾을 수 없습니다'라고 솔직하게 답변하세요.

Context: {context}

질문: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. [핵심] 검색된 문서 3개를 하나의 깔끔한 텍스트로 뭉쳐주는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. LCEL(LangChain Expression Language) 체인 조립! (버그 완벽 우회)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() # 결과를 딕셔너리가 아닌 깔끔한 문자열로 바로 뽑아줍니다.
)

print("\n✅ KISA 가이드라인 보안 챗봇이 켜졌습니다! (종료하려면 '종료' 입력)")
print("-" * 50)

# 7. 무한 채팅 루프
while True:
    query = input("\n질문해 주세요: ")
    
    if query == "종료":
        print("챗봇을 종료합니다.")
        break
        
    if not query.strip():
        continue
        
    print("답변을 생각하는 중입니다...\n")
    
    # 질문을 던지고 결과 받기 (코드가 훨씬 간결해집니다)
    result = qa_chain.invoke(query)
    
    print("🤖 챗봇의 답변:")
    print(result)
    print("-" * 50)