FROM docker.io/python:3.10.13-slim-bookworm

RUN mkdir /app

WORKDIR /app

COPY ./requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt

COPY ./src /app/src
COPY ./chainlit.md /app

EXPOSE 8000

ENTRYPOINT ["chainlit", "run", "./src/app.py", "-w"]
