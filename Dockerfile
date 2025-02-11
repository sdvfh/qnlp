FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

COPY ./code ./code

RUN apt update \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# CMD ["python", "code/run.py"]
