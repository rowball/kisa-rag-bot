from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("저장된 뇌(벡터 DB)를 깨우는 중입니다...")

# 1. DB에 넣을 때 사용했던 번역기(임베딩 모델)를 똑같이 세팅합니다.
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# 2. 방금 우리가 만든 chroma_db 폴더를 불러옵니다.
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 3. KISA 가이드라인에 있을 법한 질문을 던져봅니다.
# (이 부분의 질문을 나중에 마음대로 바꿔가며 테스트해 볼 수 있습니다)
question = "SQL 삽입 공격을 방어하려면 어떻게 코드를 짜야 해?"
print(f"\n나의 질문: {question}\n")

# 4. DB에서 내 질문과 가장 의미가 비슷한 문단 3개를 찾아옵니다.
docs = vector_db.similarity_search(question, k=3)

print("💡 가이드라인에서 찾아낸 가장 관련성 높은 문단들:")
for i, doc in enumerate(docs):
    print(f"\n--- [검색된 결과 {i+1}] ---")
    print(doc.page_content)
    # 어느 페이지에서 가져왔는지 출처도 함께 확인합니다.
    print(f"(출처: 페이지 {doc.metadata.get('page', '알 수 없음')})")