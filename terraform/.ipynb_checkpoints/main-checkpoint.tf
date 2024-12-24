resource "google_cloudbuild_trigger" "default" {
    provider = google
      name = "gradio-app-trigger"
      project = var.project_id
    location = var.region
    github {
        owner = "huguensjean" # Replace
       name  = "gradio-app" # Replace
       push {
       branch = "^main$"
        }
    }
      build {
          steps {
            name = "gcr.io/cloud-builders/docker"
             args = [
               "build",
               "-t",
               "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:\\$SHORT_SHA",
               ".",
             ]
         }

         steps {
           name = "gcr.io/cloud-builders/docker"
            args = [
              "push",
              "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repo_name}/gradio-app-image:\\$SHORT_SHA",
            ]
        }

        steps {
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