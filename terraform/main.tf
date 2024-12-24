terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "default" {
  provider      = google
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repo_name
  format        = "DOCKER"
}

resource "google_cloudbuild_trigger" "default" {
  provider = google
  name     = "gradio-app-trigger"
  project  = var.project_id
  location = var.region
  github {
    owner = "YOUR_GITHUB_USERNAME" # Replace
    name  = "YOUR_GITHUB_REPO_NAME" # Replace
    push {
      branch = "^main$"
    }
  }
  build {
    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "build",
        "-t",
        "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:\\$SHORT_SHA",
        ".",
      ]
    }

    step {
      name = "gcr.io/cloud-builders/docker"
      args = [
        "push",
        "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:\\$SHORT_SHA",
      ]
    }

    step {
      name = "gcr.io/cloud-builders/gcloud"
      args = [
        "run",
        "deploy",
        "gradio-app",
        "--image",
        "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:\\$SHORT_SHA",
        "--platform=managed",
        "--allow-unauthenticated",
        "--port=7860",
      ]
    }

    options {
      substitution_option = "ALLOW_LOOSE"
    }
  }
}

resource "google_cloud_run_service" "default" {
  provider = google
  name     = "gradio-app"
  location = var.region
  project  = var.project_id

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "1"
      }
    }
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:latest"
        ports {
          container_port = 7860
        }
      }
    }
  }
  traffic {
    latest_revision = true
    percent         = 100
  }
}

resource "google_project_service_identity" "artifact_registry" {
  provider = google
  project  = var.project_id
  service  = "artifactregistry.googleapis.com"
}

resource "google_project_service_identity" "cloudbuild" {
  provider = google
  project  = var.project_id
  service  = "cloudbuild.googleapis.com"
}

resource "google_project_iam_member" "cloud_build_registry_pusher" {
  project  = var.project_id
  role     = "roles/artifactregistry.writer"
  member   = "serviceAccount:${google_project_service_identity.cloudbuild.email}"
  depends_on = [google_project_service_identity.cloudbuild]
}

resource "google_project_iam_member" "cloud_run_invoker" {
  project = var.project_id
  role    = "roles/run.invoker"
  member  = "allUsers"
}

resource "google_project_iam_member" "cloud_run_deployer" {
  project    = var.project_id
  role       = "roles/run.developer"
  member     = "serviceAccount:${google_project_service_identity.cloudbuild.email}"
  depends_on = [google_project_service_identity.cloudbuild]
}

resource "google_project_iam_member" "artifact_registry_reader" {
  project    = var.project_id
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_project_service_identity.cloudbuild.email}"
  depends_on = [google_project_service_identity.cloudbuild]
}

resource "google_project_iam_member" "artifact_registry_writer" {
  project    = var.project_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_project_service_identity.cloudbuild.email}"
  depends_on = [google_project_service_identity.artifact_registry, google_project_service_identity.cloudbuild]
}

resource "google_project_iam_member" "run_service_agent_role" {
  project    = var.project_id
  role       = "roles/iam.serviceAccountUser"
  member     = "serviceAccount:${google_project_service_identity.cloudbuild.email}"
  depends_on = [google_project_service_identity.cloudbuild]
}