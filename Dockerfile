FROM python:3.11

WORKDIR /app

COPY requirements.txt .env .

RUN pip install -r requirements.txt

COPY . .


CMD ["python", "-u", "server.py"]
