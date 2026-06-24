import os
# PDF용 로더로 변경!
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 텍스트 데이터 불러오기 (PDF)
print("PDF 데이터를 불러오는 중입니다... (페이지 수가 많아 수십 초 정도 걸릴 수 있습니다)")
# 파일 이름을 kisa_guide.pdf 로 바꿨다고 가정합니다.
loader = PyPDFLoader("kisa_guide.pdf") 
documents = loader.load()

# 2. 데이터를 AI가 이해하기 좋은 크기(Chunk)로 잘게 쪼개기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"총 {len(chunks)}개의 조각으로 성공적으로 분할되었습니다.")

# 3. 한국어 특화 무료 오픈소스 임베딩 모델 로드 (HuggingFace)
print("오픈소스 임베딩 모델을 로드하는 중입니다...")
model_name = "jhgan/ko-sroberta-multitask" 
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 4. 쪼개진 텍스트를 벡터(숫자)로 변환하여 DB에 저장하기
persist_directory = "./chroma_db"
print("벡터 DB에 데이터를 적재하고 있습니다. (시간이 조금 걸립니다)")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

vector_db.persist()
print("✅ 벡터 DB 업데이트가 완벽하게 완료되었습니다!")