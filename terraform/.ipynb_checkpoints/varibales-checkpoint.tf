 variable "project_id" {
   description = "The ID of the Google Cloud project"
 }

 variable "region" {
   description = "The region to deploy resources in"
   default     = "us-central1"
 }
  variable "artifact_repo_name" {
   description = "The name of the artifact registry repo"
    default     = "gradio-app-repo"
 }