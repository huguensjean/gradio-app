import apache_beam as beam
import datetime
import json
import logging
import uuid
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import firestore
import os
import mimetypes
from google import genai
from google.genai import types
from google.cloud import storage
import re
import sys

# Constants for Gemini and Pub/Sub Topics
PROJECT_ID = os.getenv("PROJECT_ID", "")  # Default to provided value
REGION = os.getenv("REGION", "us-central1")  # Default to provided value
INPUT_SUBSCRIPTION = os.getenv("INPUT_SUBSCRIPTION", "")  # Fallback if not set
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "")  # Fallback if not set
BUCKET_NAME = os.getenv("BUCKET_NAME", "")  # Fallback if not set
FIRESTORE_COLLECTION = ""  # Default if not set

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

gemini_client = genai.Client(
    vertexai=True, project=PROJECT_ID, location=REGION
)

def extract_gcs_url(text):
    """Extracts GCS URLs from a text string."""
    try:
      if not text:
          logging.debug("No text provided to extract GCS URL from.")
          return None
      # Regular expression to find URLs that start with 'https://storage.googleapis.com/'
      pattern = re.compile(r'https://storage\.googleapis\.com/[a-zA-Z0-9._-]+/[a-zA-Z0-9._/-]+')
      match = pattern.search(text)
      if match:
          logging.debug(f"Extracted GCS URL: {match.group(0)}")
          return match.group(0)
      else:
        logging.debug("No GCS URL found in text.")
        return None
    except Exception as e:
      logging.error(f"Error extracting GCS URL: {e}")
      return None


def download_file_from_gcs(gcs_url):
    """Downloads a file from GCS and returns bytes, with error handling."""
    try:
        if not gcs_url:
            logging.warning("No GCS URL provided for download.")
            return None, None

        storage_client = storage.Client()
        bucket_name = gcs_url.split("/")[3]
        blob_name = "/".join(gcs_url.split("/")[4:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        file_bytes = blob.download_as_bytes()
        mime_type, _ = mimetypes.guess_type(gcs_url)
        logging.info(f"Successfully downloaded file from GCS: {gcs_url}")
        return file_bytes, mime_type
    except Exception as e:
        logging.error(f"Error downloading file from GCS: {gcs_url}. Exception: {e}")
        return None, None


def generate_gemini_response(message):
    """Generates a response from Gemini, with more detailed error handling."""
    text = message.get("text")
    if not text:
        logging.warning("No text in message for Gemini processing.")
        return None
    logging.debug(f"Processing message for Gemini: {text[:50]}...") # logging only the first 50 chars
    file_bytes = None
    file_mime = None

    return {"prediction": "Hello World.", "original_message": message}



def format_for_firestore(data):
    """Formats the Dataflow output for Firestore, and catches exception during formatting."""
    try:
        if not data:
            logging.warning("No data for Firestore formatting")
            return None
        formatted_data = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "data": data,
            "message_id": str(uuid.uuid4())
        }
        logging.debug(f"Formatted data for firestore. Message ID: {formatted_data['message_id']}")
        return formatted_data
    except Exception as e:
        logging.error(f"Error Formatting data for firestore: {e}")
        return None

class FirestoreWriter(beam.DoFn):
    """Writes data to Firestore, catching any exceptions during the write."""
    def __init__(self, collection):
        self.collection = collection

    def process(self, data):
      try:
        firestore_client = firestore.Client()  # Create client in process
        doc_ref = firestore_client.collection(self.collection).document(data["message_id"])
        doc_ref.set(data)
        logging.debug(f"Successfully wrote to firestore. Message ID: {data['message_id']}")
        yield f"Wrote to firestore: {data['message_id']}"
      except Exception as e:
        logging.error(f"Error writing to firestore: {e}")
        yield f"Error Writing to firestore: {e}"

def run_pipeline(input_subscription, output_topic, gcp_project_id, gcp_region):
    import uuid
    import datetime
    import json
    
    """Runs the Dataflow pipeline, with detailed error messages."""
    try:
        with beam.Pipeline(options=PipelineOptions(
            streaming=True,
            temp_location=f'gs://{BUCKET_NAME}/tmp',
            staging_location=f'gs://{BUCKET_NAME}/staging',
        )
        ) as pipeline:
            messages = (
                pipeline
                | "Read from Pub/Sub"
                >> beam.io.ReadFromPubSub(subscription=input_subscription)
                | "Parse JSON"
                >> beam.Map(lambda x: json.loads(x) if x else None)
                | "Filter None JSON" >> beam.Filter(lambda x: x is not None) # Ensure no malformed json makes it through
            )
            logging.info(f"Read messages from Pub/Sub subscription: {input_subscription}")

            predictions = (
                messages
                | "Call Gemini API"
                >> beam.Map(generate_gemini_response)
            )

            valid_predictions = predictions | "Filter None Gemini Response" >> beam.Filter(lambda x: x is not None)
            logging.info(f"Processed messages using Gemini API.")

            # Output to Pub/Sub
            (
                valid_predictions
                | "Format Output for Pub/Sub"
                >> beam.Map(lambda x: {"text": x['prediction'], "original_message_id": x['original_message']['message_id'], 'sender': 'Dataflow', 'app_name': 'Dataflow',
                                       'message_id': str(uuid.uuid4()), "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
                | "Convert to String"
                >> beam.Map(lambda x: json.dumps(x).encode('utf-8'))
                | "Output to Pub/Sub"
                >> beam.io.WriteToPubSub(topic=output_topic, with_attributes=True)
            )
            logging.info(f"Outputted Gemini responses to Pub/Sub topic: {output_topic}")


            # Output to Firestore
            (valid_predictions
             | "Format for Firestore"
             >> beam.Map(format_for_firestore)
             | "Filter None Firestore" >> beam.Filter(lambda x: x is not None)
             | "Write to Firestore"
             >> beam.ParDo(FirestoreWriter(FIRESTORE_COLLECTION))
             )
            logging.info(f"Outputted Gemini responses to Firestore collection: {FIRESTORE_COLLECTION}")

    except Exception as e:
      logging.error(f"Pipeline Error: {e}")
      return None

if __name__ == "__main__":
    run_pipeline(INPUT_SUBSCRIPTION, OUTPUT_TOPIC, PROJECT_ID, REGION)
