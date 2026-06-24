# 1. 파이썬이 설치된 가벼운 리눅스(slim)를 베이스로 가져옵니다.
FROM python:3.11-slim

# 2. 도커 컨테이너 안의 작업 폴더를 /app 으로 설정합니다.
WORKDIR /app

# 3. 필수 C++ 빌드 도구를 설치합니다 (ChromaDB 구동에 필요)
# 캐시 폴더까지 깔끔하게 지워 도커 이미지 용량을 몇십 MB 더 다이어트합니다.
RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. 패키지 명세서를 복사하고 설치합니다.
# requirements.txt 맨 첫 줄에 적어둔 CPU 인덱스 규칙을 따라 아주 가볍게 설치됩니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 챗봇 실행 코드와 만들어둔 '뇌(DB)' 폴더만 컨테이너로 복사합니다.
COPY chat_with_gemini.py .
COPY chroma_db/ ./chroma_db/

# 6. 컨테이너가 켜지면 챗봇 파이썬 파일을 실행합니다.
CMD ["python", "chat_with_gemini.py"]