FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY train.py predict.py ./
RUN python train.py

EXPOSE 9696
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
