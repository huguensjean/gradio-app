FROM us-central1-docker.pkg.dev/automl-migration-test/gradio-app-repo/gradio-app-base-image:latest

WORKDIR /app

RUN git clone --branch main https://github.com/GoogleCloudPlatform/vertex-ai-samples.git

RUN git clone --quiet --branch=main --depth=1 \
     https://github.com/google-research/big_vision big_vision_repo

COPY utils/gcp_utils.py .
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]