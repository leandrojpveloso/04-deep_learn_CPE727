FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY script.py /app/script.py

ENTRYPOINT ["python3", "/app/script.py"]
