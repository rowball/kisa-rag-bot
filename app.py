import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 환경변수(.env) 로드
load_dotenv()

# 2. 웹사이트 탭 디자인 설정
st.set_page_config(page_title="KISA 보안 챗봇", page_icon="🛡️")
st.title("🛡️ KISA 보안 가이드 챗봇")
st.caption("한국인터넷진흥원 가이드라인을 바탕으로 답변합니다.")

# 3. RAG 뇌(DB + AI) 및 가드레일 세팅
@st.cache_resource
def load_rag_chain():
    # 임베딩 및 벡터 DB 로드
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask"
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Gemini LLM 로드
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # -----------------------------------------------------------------
    # [포트폴리오용 추가] 1단계: 가드레일 (비용 절감 및 주제 이탈 방지)
    # -----------------------------------------------------------------
    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 입력된 질문이 '정보보안, KISA, 해킹, IT 인프라, 개인정보보호, 컴퓨터'와 관련된 질문인지 판별하는 판사입니다. 관련이 있다면 'YES', 전혀 무관하다면 'NO'만 대답하세요."),
        ("human", "{input}")
    ])
    guardrail_chain = guardrail_prompt | llm | StrOutputParser()

    # -----------------------------------------------------------------
    # 2단계: 메인 RAG 파이프라인
    # -----------------------------------------------------------------
    system_prompt = (
        "당신은 KISA 보안 가이드라인을 안내하는 전문가입니다. "
        "주어진 문맥(Context)을 바탕으로 친절하고 정확하게 답변하세요. "
        "문맥에 없는 내용은 '제공된 문서에서 찾을 수 없습니다'라고 답변하세요.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return guardrail_chain, rag_chain # 가드레일과 메인 체인 둘 다 반환

# AI 뇌 장착
guardrail_chain, rag_chain = load_rag_chain()

# 4. 채팅 기록 저장소 만들기
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! KISA 보안 가이드에 대해 무엇이든 물어보세요."}]

# 5. 기존 채팅 기록 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 사용자 입력 창 및 로직 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("질문을 분석 중입니다..."):
            # 1. 가드레일 통과 여부 확인 (빠르고 저렴한 검사)
            is_secure_query = guardrail_chain.invoke({"input": prompt})
            
        if "NO" in is_secure_query.upper():
            # 엉뚱한 질문이면 메인 체인을 실행하지 않고 차단 (토큰/DB비용 절약!)
            block_msg = "🔒 **보안 및 IT 관련 질문에만 답변할 수 있습니다.** KISA 가이드라인에 대해 다시 질문해 주세요!"
            st.markdown(block_msg)
            st.session_state.messages.append({"role": "assistant", "content": block_msg})
        else:
            # 보안 질문이면 정상적으로 DB 검색 및 답변 생성
            with st.spinner("문서에서 정답을 찾는 중..."):
                answer = rag_chain.invoke(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})