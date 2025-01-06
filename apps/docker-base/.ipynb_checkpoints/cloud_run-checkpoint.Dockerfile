FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/cloud-run-gradio-app-base:latest

WORKDIR /usr/src/app

COPY . .

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python3", "app.py"]