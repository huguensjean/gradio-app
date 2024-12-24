FROM us-central1-docker.pkg.dev/automl-migration-test/gradio-app-repo/gradio-app-image:latest

WORKDIR /app

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]