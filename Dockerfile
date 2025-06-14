FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y procps net-tools && apt-get clean

COPY . .


EXPOSE 8265 10001 6379

CMD [ "python","train.py" ]