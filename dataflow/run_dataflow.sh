#!/bin/bash

# Set your Google Cloud project ID and region
PROJECT_ID=""
REGION=""

# Set your Google Cloud Storage bucket for staging and temporary files
GCS_BUCKET=""

# Set the location to your python script
DATAFLOW_SCRIPT="dataflow_pipeline.py"

# Ensure that PROJECT_ID, REGION, and GCS_BUCKET are set
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$GCS_BUCKET" ]; then
  echo "Error: PROJECT_ID, REGION, and GCS_BUCKET must be set."
  echo "Please set these variables in the script or as environment variables."
  exit 1
fi

# Ensure that the script exists
if [ ! -f "$DATAFLOW_SCRIPT" ]; then
    echo "Error: The dataflow script '$DATAFLOW_SCRIPT' was not found."
    exit 1
fi


# Print information before running
echo "Running Dataflow pipeline with:"
echo "  Project ID: $PROJECT_ID"
echo "  Region:     $REGION"
echo "  GCS Bucket: $GCS_BUCKET"
echo "  Script:     $DATAFLOW_SCRIPT"

# Run the Dataflow pipeline
python "dataflow_pipeline.py" \
    --runner DataflowRunner \
    --project "" \
    --region "" \
    --temp_location "" \
    --staging_location ""

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Dataflow pipeline launched successfully."
else
    echo "Error: Dataflow pipeline failed to launch."
    exit 1
fi