# 1. 파이썬이 설치된 가벼운 리눅스(slim)를 베이스로 가져옵니다.
FROM python:3.11-slim

# 2. 도커 컨테이너 안의 작업 폴더를 /app 으로 설정합니다.
WORKDIR /app

# 3. 필수 C++ 빌드 도구를 설치합니다 (ChromaDB 구동에 필요)
RUN apt-get update && apt-get install -y build-essential

# 4. 방금 만든 패키지 명세서를 복사하고 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. [핵심] 챗봇 실행 코드와 만들어둔 '뇌(DB)' 폴더만 컨테이너로 복사합니다.
COPY chat_with_gemini.py .
COPY chroma_db/ ./chroma_db/

# 6. 컨테이너가 켜지면 챗봇 파이썬 파일을 실행합니다.
CMD ["python", "chat_with_gemini.py"]