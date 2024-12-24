FROM python:3.11-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    python3-dev \
    git \
    curl \
    gcc \
    python3-setuptools \
    apt-transport-https \
    lsb-release \
    openssh-client \
    gnupg \
    ca-certificates \
    && apt-get autoremove -y

# Install crcmod (improves gsutil performance).
RUN pip install -U crcmod==1.7

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y \
    google-cloud-sdk \
    google-cloud-sdk-app-engine-python \
    google-cloud-sdk-app-engine-python-extras \
    kubectl

RUN gcloud --version && kubectl version --client

# Fix for ImportError: libgthread-2.0.so.0: cannot open shared object file.
# For fixing ImportError: libGL.so.1: cannot open shared object file.
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN git clone --branch main https://github.com/GoogleCloudPlatform/vertex-ai-samples.git

RUN git clone --quiet --branch=main --depth=1 \
     https://github.com/google-research/big_vision big_vision_repo
