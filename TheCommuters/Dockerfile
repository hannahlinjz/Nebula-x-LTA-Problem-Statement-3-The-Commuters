FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8080

CMD ["bash", "-lc", "streamlit run app/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]
