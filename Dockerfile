FROM us-central1-docker.pkg.dev/automl-migration-test/gradio-app-repo/gradio-app-base-image:latest

WORKDIR /app

COPY utils/gcp_utils.py .
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]