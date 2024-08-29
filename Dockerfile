FROM python:3.10.6

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD uvicorn lateguru_ml.api.fast:app --host 0.0.0.0 --port 8080 --reload
