"""Pubsub app with APScheduler for continuous polling, plus local caching for topics/subscriptions."""

import concurrent.futures
import datetime
import hashlib
import json
import logging
import mimetypes
import os
import sys
import threading
import time
import uuid

from apscheduler.schedulers import background
from common_utils import create_publisher
from common_utils import create_subscriber
from common_utils import create_subscription
from common_utils import create_topic
from common_utils import project_id
from common_utils import publish_message
from common_utils import pull_messages
import fastapi
from google import auth
from google.api_core import exceptions
from google.auth.transport import requests
from google.cloud import firestore
from google.cloud import storage
import gradio as gr
import gradiologin as gl
import uvicorn


###############################################################################
# Global Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BackgroundScheduler = background.BackgroundScheduler

# For credentials
default = auth.default
Request = requests.Request

###############################################################################
# Constants and Settings
###############################################################################
DEFAULT_APP_NAME = "Wendy"
MAX_MESSAGE_LENGTH = 500
MAX_PUBLISH_RETRIES = 4

BUCKET_NAME = os.getenv("BUCKET_NAME", "huguens-testing-model-garden")
FIRESTORE_COLLECTION = "chat_history"
MEDIA_COLLECTION = "media_files"
CREDENTIALS_PATH = "keyfile.json"

###############################################################################
# Pub/Sub Clients
###############################################################################
logging.info("Creating Pub/Sub publisher and subscriber clients...")
publisher = create_publisher()
subscriber = create_subscriber()

###############################################################################
# In-memory DB and State
###############################################################################
processed_message_ids = set()
chat_lock = threading.Lock()
APP_NAME = DEFAULT_APP_NAME

# Local caches for topics & subscriptions
available_topics = set()
available_subscriptions = set()
user_subscriptions = {}
subscribed_topics = set()

###############################################################################
# Firestore Client
###############################################################################
firestore_client = firestore.Client()
logging.info("Firestore client initialized.")


def get_firestore_client():
  return firestore.Client()


###############################################################################
# Helper Functions
###############################################################################
def upload_file_to_gcs(local_file_path: str) -> str:
  """Uploads a file to Google Cloud Storage; returns GCS URL or None."""
  try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = os.path.basename(local_file_path)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)
    blob_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{blob_name}"
    logging.info(f"File uploaded to GCS: {blob_url}")
    return blob_url
  except Exception as e:
    logging.error(f"Error uploading to GCS: {e}")
    return None


def store_media_metadata(file_id: str, gcs_url: str, mime_type: str):
  """Stores GCS URL and mime_type in Firestore for later retrieval."""
  try:
    doc_ref = firestore_client.collection(MEDIA_COLLECTION).document(file_id)
    doc_ref.set({"gcs_url": gcs_url, "mime_type": mime_type})
    logging.debug(f"Stored metadata for file_id: {file_id}")
  except Exception as e:
    logging.error(f"Error storing media metadata: {e}")


def get_media_html(file_id: str) -> str:
  """Retrieves GCS URL from Firestore and generates HTML for image/video/file link."""
  try:
    doc_ref = firestore_client.collection(MEDIA_COLLECTION).document(file_id)
    doc = doc_ref.get()

    if not doc.exists:
      logging.error(f"Media metadata not found: {file_id}")
      return "Error: File not found"

    media_data = doc.to_dict()
    gcs_url = media_data["gcs_url"]
    mime_type = media_data["mime_type"]

    if gcs_url and mime_type:
      if mime_type.startswith("video"):
        return (
            '<video controls playsinline width="500">'
            f'<source src="/get_gcs_file?file_id={file_id}" type="{mime_type}">'
            "Your browser does not support the video tag.</video>"
        )
      elif mime_type.startswith("image"):
        return (
            f'<img src="/get_gcs_file?file_id={file_id}" alt="Uploaded Image"'
            ' width="300" />'
        )
      else:
        return (
            f'<a href="/get_gcs_file?file_id={file_id}" target="_blank">'
            "File in GCS</a>"
        )
    else:
      logging.error(f"Error: GCS URL or mime_type not found for {file_id}")
      return "Error: Could not generate link"
  except Exception as e:
    logging.error(f"Error retrieving media HTML: {e}")
    return "Error generating media HTML"


def validate_message(text: str):
  """Validates the message text for length and emptiness."""
  if not text:
    return False, "Message cannot be empty."
  if len(text) > MAX_MESSAGE_LENGTH:
    return False, f"Message exceeds maximum length ({MAX_MESSAGE_LENGTH})."
  return True, None


def store_message_entire_json(topic_name: str, message_dict: dict):
  """Stores the entire incoming message JSON to Firestore,

  ensuring we preserve all fields (sender, text, etc.).
  We also add the 'topic_name' in the doc for easy retrieval.
  """
  try:
    # Make a copy so we don't mutate the original
    doc_data = dict(message_dict)
    doc_data["topic_name"] = topic_name

    # Ensure we have a message_id
    message_id = doc_data.get("message_id", str(uuid.uuid4()))

    doc_ref = firestore_client.collection(FIRESTORE_COLLECTION).document(
        message_id
    )
    doc_ref.set(doc_data)
    logging.debug(f"Inserted/Updated doc with message_id: {message_id}")
  except Exception as e:
    logging.error(f"Error storing message in Firestore: {e}")


def get_chat_history_from_db(topic_name: str):
  """Return a list of (sender, text) messages for the given topic_name from Firestore,

  ordered by the stored 'timestamp' (or 'publish_time', whichever your messages
  have).
  """
  try:
    docs = (
        firestore_client.collection(FIRESTORE_COLLECTION)
        .where("topic_name", "==", topic_name)
        .order_by("timestamp")  # or .order_by("publish_time") if you prefer
        .get()
    )
    history = []
    for doc in docs:
      d = doc.to_dict()
      sender = d["sender"]  # Defaults to "UNKNOWN" if not present
      text = d["text"]  # Defaults to "" if not present
      history.append((sender, text))
    return history
  except Exception as e:
    logging.error(f"Error getting chat history from database: {e}")
    return []


###############################################################################
# Pub/Sub Callback
###############################################################################
def message_callback(message, subscription_name: str):
  """Callback for handling messages pulled from Pub/Sub.

  Reads the entire JSON, preserves all fields, then writes them to Firestore
  under the appropriate topic name.
  """
  global processed_message_ids
  try:
    # Decode entire JSON
    data_str = message.data.decode("utf-8")
    data = json.loads(data_str)

    # Determine the topic from the subscription
    subscription_path = subscriber.subscription_path(
        project_id, subscription_name
    )
    sub = subscriber.get_subscription(
        request={"subscription": subscription_path}
    )
    topic_name = sub.topic.split("/")[-1]

    if not topic_name:
      logging.error(
          f"Could not find topic name for subscription: {subscription_name}"
      )
      message.ack()
      return

    # Check if already processed
    message_id = data.get("message_id")
    if message_id and message_id in processed_message_ids:
      logging.info(f"Skipping already processed message {message_id}")
      message.ack()
      return

    # Store the message (all fields) in Firestore
    with chat_lock:
      store_message_entire_json(topic_name, data)
      if message_id:
        processed_message_ids.add(message_id)

    message.ack()

  except json.JSONDecodeError as e:
    logging.error(f"Error decoding message data: {e}, nacking.")
    message.nack()
  except Exception as e:
    logging.error(f"Error in message_callback: {e}, nacking.")
    message.nack()


###############################################################################
# Publish with Retry
###############################################################################
def publish_message_with_retry(
    publisher_client, topic_path, msg_data, max_retries=4
):
  """Publishes a message with retry logic."""
  for attempt in range(max_retries):
    try:
      published = publish_message(publisher_client, topic_path, msg_data)
      if published:
        return True
    except exceptions.GoogleAPIError as e:
      logging.error(f"Publish attempt {attempt+1} failed: {e}")
      time.sleep(2**attempt)
    except Exception as e:
      logging.error(f"Unexpected error: {e}")
      time.sleep(2**attempt)
  logging.error("Failed to publish after multiple retries.")
  return False


###############################################################################
# Update Local Cache with GCP
###############################################################################
def update_available_topics():
  """Updates the local cache of available topics from GCP."""
  global available_topics
  try:
    topic_list = publisher.list_topics(
        request={"project": f"projects/{project_id}"}
    )
    remote_topic_names = {t.name.split("/")[-1] for t in topic_list.topics}

    available_topics = available_topics.union(remote_topic_names)
    logging.info(f"Updated local topics: {available_topics}")
    return sorted(list(available_topics))
  except Exception as e:
    logging.error(f"Error getting topics: {e}")
    return sorted(list(available_topics))


def update_available_subscriptions():
  """Updates the local cache of available subscriptions from GCP."""
  global available_subscriptions
  try:
    subs = subscriber.list_subscriptions(
        request={"project": f"projects/{project_id}"}
    )
    remote_sub_names = {s.name.split("/")[-1] for s in subs.subscriptions}

    available_subscriptions = remote_sub_names
    logging.info(f"Updated local subscriptions: {available_subscriptions}")
    return sorted(list(available_subscriptions))
  except Exception as e:
    logging.error(f"Error getting subscriptions: {e}")
    return sorted(list(available_subscriptions))


###############################################################################
# Topic Creation & Deletion
###############################################################################
def create_new_topic(topic_name_input: str):
  """Creates a new Pub/Sub topic."""
  global available_topics
  if not topic_name_input:
    return "Topic name cannot be empty", gr.update()

  logging.info(
      f"create_new_topic called for '{topic_name_input}' in project"
      f" '{project_id}'"
  )
  try:
    topic_path = create_topic(publisher, project_id, topic_name_input)
    logging.info(f"Created topic in GCP: {topic_path}")

    available_topics.add(topic_name_input)
    logging.debug(f"Local topics after create: {available_topics}")

    updated_list = update_available_topics()
    if topic_name_input not in updated_list:
      final_list = sorted(list(set(updated_list) | {topic_name_input}))
    else:
      final_list = updated_list

    return (
        f"Topic '{topic_name_input}' created.",
        gr.update(choices=final_list, value=topic_name_input),
    )
  except Exception as e:
    logging.error(f"Error creating topic '{topic_name_input}': {e}")
    return f"Error creating topic: {e}", gr.update()


def delete_topic(topic_name_input: str):
  """Deletes a Pub/Sub topic."""
  global available_topics
  if not topic_name_input:
    return "No topic selected to delete.", gr.update()

  logging.info(
      f"delete_topic called for '{topic_name_input}' in project '{project_id}'"
  )

  try:
    topic_path = publisher.topic_path(project_id, topic_name_input)
    publisher.delete_topic(request={"topic": topic_path})
    logging.info(f"Deleted topic in GCP: {topic_path}")

    available_topics.discard(topic_name_input)
    logging.debug(f"Local topics after delete: {available_topics}")

    updated_list = update_available_topics()
    return f"Topic '{topic_name_input}' deleted.", gr.update(
        choices=updated_list, value=None
    )

  except exceptions.NotFound:
    msg = f"Topic '{topic_name_input}' not found or already deleted in GCP."
    logging.error(msg)
    available_topics.discard(topic_name_input)
    updated_list = update_available_topics()
    return msg, gr.update(choices=updated_list, value=None)

  except Exception as e:
    msg = f"Error deleting topic '{topic_name_input}': {e}"
    logging.error(msg)
    return msg, gr.update()


###############################################################################
# Subscription Creation & Deletion
###############################################################################
def create_new_subscription(topic_name, subscription_name_input, user_id):
  """Creates a Pub/Sub subscription with a unique name per user."""
  global available_subscriptions

  if not subscription_name_input or not topic_name:
    return (
        "Subscription name or Topic cannot be empty.",
        update_available_subscriptions(),
    )

  unique_subscription_name = (  # ADDED topic name to sub name
      f"{topic_name}-{subscription_name_input}-{user_id}"
  )
  logging.info(
      f"create_new_subscription called for '{unique_subscription_name}' on"
      f" topic '{topic_name}'"
  )

  try:
    subscription_path = create_subscription(
        subscriber, project_id, topic_name, unique_subscription_name
    )
    logging.info(f"Created subscription in GCP: {subscription_path}")

    available_subscriptions.add(unique_subscription_name)
    logging.debug(f"Local subs after create: {available_subscriptions}")

    updated_subs = update_available_subscriptions()
    if unique_subscription_name not in updated_subs:
      final_sub_list = sorted(
          list(set(updated_subs) | {unique_subscription_name})
      )
    else:
      final_sub_list = updated_subs

    return f"Subscription '{unique_subscription_name}' created.", final_sub_list
  except Exception as e:
    logging.error(
        f"Error creating subscription '{unique_subscription_name}': {e}"
    )
    return f"Error creating subscription: {e}", update_available_subscriptions()


def delete_subscription(subscription_name_input: str):
  """Deletes a Pub/Sub subscription."""
  global available_subscriptions, user_subscriptions
  if not subscription_name_input:
    return "No subscription selected to delete.", gr.update()

  logging.info(
      f"delete_subscription called for '{subscription_name_input}' in project"
      f" '{project_id}'"
  )

  try:
    subscription_path = subscriber.subscription_path(
        project_id, subscription_name_input
    )
    subscriber.delete_subscription(request={"subscription": subscription_path})
    logging.info(f"Deleted subscription in GCP: {subscription_path}")

    available_subscriptions.discard(subscription_name_input)
    if subscription_name_input in user_subscriptions:
      del user_subscriptions[subscription_name_input]

    updated_subs = update_available_subscriptions()
    return f"Subscription '{subscription_name_input}' deleted.", gr.update(
        choices=updated_subs, value=None
    )

  except exceptions.NotFound:
    msg = (
        f"Subscription '{subscription_name_input}' not found or already deleted"
        " in GCP."
    )
    logging.error(msg)
    available_subscriptions.discard(subscription_name_input)
    if subscription_name_input in user_subscriptions:
      del user_subscriptions[subscription_name_input]
    updated_subs = update_available_subscriptions()
    return msg, gr.update(choices=updated_subs, value=None)

  except Exception as e:
    msg = f"Error deleting subscription '{subscription_name_input}': {e}"
    logging.error(msg)
    return msg, gr.update()


def create_subscription_with_user_id(
    topic_name_input, sub_name_input, request: gr.Request
):
  """UI handler that calls create_new_subscription(...) with a hashed user ID."""
  user = gl.get_user(request)
  user_id = hashlib.md5(user["name"].encode()).hexdigest()[:6]
  message, subs = create_new_subscription(
      topic_name_input, sub_name_input, user_id
  )

  # Get the final name of the subscription to ensure it is correct
  unique_sub = f"{topic_name_input}-{sub_name_input}-{user_id}"
  new_sub_value = unique_sub if unique_sub in subs else None

  return (
      message,
      gr.update(choices=subs, value=new_sub_value),
      gr.update(value=[topic_name_input]),  # Forces topic update
  )


###############################################################################
# Sending Messages
###############################################################################
def send_message(
    message_input, app_name_input, selected_topic, selected_subscription
):
  """Sends a message to the selected Pub/Sub topic (including optional files)."""
  logging.info(
      f"send_message called for topic: {selected_topic},"
      f" sub:{selected_subscription}"
  )

  text = message_input.get("text")
  files = message_input.get("files", [])

  # ------------------------------------------------------------------
  # FIX: Ensure app_name_input is never empty so 'sender' is populated.
  # ------------------------------------------------------------------
  if not app_name_input or not app_name_input.strip():
    app_name_input = "UnknownUser"

  # Fail-safe: if no topic selected
  if not selected_topic:
    return (
        [("System", "Please select a topic first!")],
        gr.MultimodalTextbox(value=None, interactive=True),
        gr.Dropdown(choices=update_available_topics(), interactive=True),
        gr.Dropdown(choices=update_available_subscriptions(), interactive=True),
    )

  # Fail-safe: if no subscription selected
  if not selected_subscription:
    return (
        [("System", "Please select a subscription first!")],
        gr.MultimodalTextbox(value=None, interactive=True),
        gr.Dropdown(
            choices=update_available_topics(),
            value=selected_topic,
            interactive=True,
        ),
        gr.Dropdown(choices=update_available_subscriptions(), interactive=True),
    )

  # Fetch topic from GCP to check for consistency
  try:
    subscription_path = subscriber.subscription_path(
        project_id, selected_subscription
    )
    sub = subscriber.get_subscription(
        request={"subscription": subscription_path}
    )
    expected_topic = sub.topic.split("/")[-1]
  except Exception as e:
    logging.error(f"Error fetching subscription details: {e}")
    return (
        [("System", f"Error: Could not verify subscription. {e}")],
        gr.MultimodalTextbox(value=None, interactive=True),
        gr.Dropdown(
            choices=update_available_topics(),
            value=selected_topic,
            interactive=True,
        ),
        gr.Dropdown(
            choices=update_available_subscriptions(),
            value=selected_subscription,
            interactive=True,
        ),
    )

  if expected_topic != selected_topic:
    logging.warning(
        f"User tried sending message to topic '{selected_topic}' with"
        f" subscription '{selected_subscription}', but subscription is routed"
        f" for '{expected_topic}'."
    )
    return (
        [(
            "System",
            "Error: The chosen subscription does not match the selected topic!",
        )],
        gr.MultimodalTextbox(value=None, interactive=True),
        gr.Dropdown(
            choices=update_available_topics(),
            value=selected_topic,
            interactive=True,
        ),
        gr.Dropdown(
            choices=update_available_subscriptions(),
            value=selected_subscription,
            interactive=True,
        ),
    )

  # If no message text/files, do nothing
  if not text and not files:
    return (
        get_chat_history_from_db(selected_topic),
        gr.MultimodalTextbox(value=None, interactive=True),
        gr.Dropdown(
            choices=update_available_topics(),
            value=selected_topic,
            interactive=True,
        ),
        gr.Dropdown(
            choices=update_available_subscriptions(),
            value=selected_subscription,
            interactive=True,
        ),
    )

  # Validate text if present
  if text:
    is_valid, error_message = validate_message(text)
    if not is_valid:
      logging.error(error_message)
      return (
          get_chat_history_from_db(selected_topic),
          gr.MultimodalTextbox(value=None, interactive=True),
          gr.Dropdown(
              choices=update_available_topics(),
              value=selected_topic,
              interactive=True,
          ),
          gr.Dropdown(
              choices=update_available_subscriptions(),
              value=selected_subscription,
              interactive=True,
          ),
      )

  timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
      timespec="microseconds"
  )
  message_id = str(uuid.uuid4())
  topic_path = publisher.topic_path(project_id, selected_topic)
  combined_text = text if text else ""

  def background_send():
    """Background logic for sending user message response to Pub/Sub."""
    nonlocal combined_text
    with chat_lock:
      # Upload any files to GCS
      for local_file_path in files:
        if os.path.exists(local_file_path):
          try:
            gcs_url = upload_file_to_gcs(local_file_path)
            mime_type, _ = mimetypes.guess_type(local_file_path)
            file_id = str(uuid.uuid4())
            store_media_metadata(file_id, gcs_url, mime_type)
            combined_text += f"<br>{get_media_html(file_id)}"
          except Exception as e:
            logging.error(f"Error uploading/storing {local_file_path}: {e}")
            combined_text += f"<br>Error uploading file: {local_file_path}"
        else:
          combined_text += f"<br>File not found: {local_file_path}"

      if combined_text:
        # Publish user message
        msg_data = {
            "sender": app_name_input,
            "text": combined_text,
            "timestamp": timestamp,
            "message_id": message_id,
            "app_name": app_name_input,
        }
        success = publish_message_with_retry(
            publisher, topic_path, msg_data, MAX_PUBLISH_RETRIES
        )
        if not success:
          logging.error("Failed to publish user message after retries.")
          return

        # Also store it in Firestore
        store_message_entire_json(selected_topic, msg_data)

  # Launch the background task
  with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.submit(background_send)

  # Return immediately; UI updates from polling
  return (
      get_chat_history_from_db(selected_topic),
      gr.MultimodalTextbox(value=None, interactive=True),
      gr.Dropdown(
          choices=update_available_topics(),
          value=selected_topic,
          interactive=True,
      ),
      gr.Dropdown(
          choices=update_available_subscriptions(),
          value=selected_subscription,
          interactive=True,
      ),
  )


###############################################################################
# Polling + Streaming
###############################################################################
def poll_updates(selected_topic, chat_output):
  """Pulls updated chat history for the currently-selected topic."""
  with chat_lock:
    if selected_topic:
      sorted_history = get_chat_history_from_db(selected_topic)
      return gr.update(value=sorted_history)
    else:
      return gr.update(value=[])


def subscribe_to_messages(topic_name, subscription_name):
  """Sync streaming pull for the subscription, invoked in a daemon thread.

  Calls message_callback for each incoming message.
  """
  subscription_path = subscriber.subscription_path(
      project_id, subscription_name
  )

  if subscription_name not in available_subscriptions:
    logging.warning(
        f"Subscription {subscription_name} was not in available_subscriptions."
        " Skipping streaming pull."
    )
    return

  streaming_pull_future = pull_messages(
      subscriber,
      subscription_path,
      callback=lambda msg: message_callback(msg, subscription_name),
  )
  logging.info(f"Streaming messages from {subscription_path}...")

  try:
    streaming_pull_future.result()  # block forever
  except (KeyboardInterrupt, SystemExit):
    streaming_pull_future.cancel()
    logging.info("Stopped streaming pull.")


def create_and_subscribe_to_topic(
    topic_name, subscription_name, request: gr.Request = None
):
  """Ensures that subscription_name is mapped to topic_name and starts streaming."""
  logging.info(
      f"create_and_subscribe_to_topic for topic='{topic_name}',"
      f" subscription='{subscription_name}'"
  )

  # Check if topic exists, else create
  topic_path = publisher.topic_path(project_id, topic_name)
  try:
    publisher.get_topic(request={"topic": topic_path})
  except exceptions.NotFound:
    create_topic(publisher, project_id, topic_name)
    logging.info(f"Created topic: {topic_name}")

  # Check if sub exists, else create
  subscription_path = subscriber.subscription_path(
      project_id, subscription_name
  )
  try:
    subscriber.get_subscription(request={"subscription": subscription_path})
  except exceptions.NotFound:
    create_subscription(subscriber, project_id, topic_name, subscription_name)
    logging.info(f"Created subscription: {subscription_name}")

  # Start streaming pull in a daemon thread
  thread = threading.Thread(
      target=subscribe_to_messages,
      args=(topic_name, subscription_name),
      daemon=True,
  )
  thread.start()
  logging.info(
      f"Thread started for sub: {subscription_name} on topic: {topic_name}"
  )


###############################################################################
# Unique Subscriptions Per User
###############################################################################
def create_default_subscription_with_user_id(request: gr.Request):
  """Creates a unique subscription for each topic for the current user."""
  user = gl.get_user(request)
  user_id = hashlib.md5(user["name"].encode()).hexdigest()[:6]

  update_available_topics()
  update_available_subscriptions()
  logging.info(f"Available topics on startup: {available_topics}")
  logging.info(f"Available subscriptions on startup: {available_subscriptions}")

  for topic in available_topics:
    sub_name = f"{topic}-sub-{user_id}"
    create_and_subscribe_to_topic(topic, sub_name, request)


def subscribe_to_selected_topics(topics, user_id):
  """Creates new subscriptions for selected topics, if they don't exist yet."""
  global subscribed_topics
  new_topics = set(topics) - subscribed_topics
  for t in new_topics:
    sub_name = f"{t}-sub-{user_id}"
    create_and_subscribe_to_topic(t, sub_name)
  subscribed_topics.update(new_topics)


###############################################################################
# APScheduler + FastAPI + Gradio
###############################################################################
scheduler = BackgroundScheduler()


def keep_polling_db():
  """Called by APScheduler every second, triggers a UI update by calling poll_updates."""
  global active_chat_output
  global active_topic_dropdown

  if active_chat_output and active_topic_dropdown:
    new_history = poll_updates(active_topic_dropdown.value, active_chat_output)
    if new_history:
      active_chat_output.value = new_history.value


scheduler.add_job(keep_polling_db, "interval", seconds=1)
scheduler.start()
logging.info("APScheduler background job started.")

FastAPI = fastapi.FastAPI
app = FastAPI()
logging.info("FastAPI app initialized.")


@app.get("/get_gcs_file")
async def get_gcs_file(file_id: str):
  """Serves GCS files via the app, authorized with service account.

  Allows inline viewing for images/videos.
  """
  try:
    storage_client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
    logging.info(f"Fetching file_id: {file_id} from /get_gcs_file")

    doc_ref = firestore_client.collection(MEDIA_COLLECTION).document(file_id)
    doc = doc_ref.get()
    if not doc.exists:
      raise fastapi.HTTPException(status_code=404, detail="File not found")

    media_data = doc.to_dict()
    gcs_url = media_data["gcs_url"]
    mime_type = media_data["mime_type"]
    if not gcs_url or not mime_type:
      raise fastapi.HTTPException(
          status_code=404, detail="GCS URL or mime type not found"
      )

    bucket_name = gcs_url.split("/")[3]
    blob_name = "/".join(gcs_url.split("/")[4:])
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    file_content = blob.download_as_bytes()

    return fastapi.Response(content=file_content, media_type=mime_type)

  except Exception as e:
    logging.error(f"Error serving file from GCS: {e}")
    raise fastapi.HTTPException(
        status_code=500, detail=f"Error serving file: {e}"
    )


# Gradio + Google Login
gl.register(
    name="google",
    server_metadata_url=(
        "https://accounts.google.com/.well-known/openid-configuration"
    ),
    client_id=(
        "6924728003-pi669mkcima7pqpdcmce5h134mnv490f.apps.googleusercontent.com"
    ),
    client_secret="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",
    client_kwargs={"scope": "openid email profile"},
)
logging.info("Google login registered.")


def show_user(request: gr.Request):
  """Shows the logged-in user's name in the Gradio UI."""
  user = gl.get_user(request)
  return gr.update(value=user["name"])


###############################################################################
# Gradio UI
###############################################################################
active_chat_output = None
active_topic_dropdown = None

with gr.Blocks(css="#chatbot {overflow:auto; height:800px;}") as demo:
  # A simple logout button
  gl.LogoutButton("Logout")

  # Displays the current logged-in user (read-only)
  app_name_input = gr.Textbox(
      label="User Name", value=DEFAULT_APP_NAME, interactive=False
  )

  with gr.Row():
    with gr.Column():
      with gr.Row():
        topic_dropdown = gr.Dropdown(
            label="Select a topic", choices=[], interactive=True, scale=2
        )
        delete_topic_button = gr.Button("Delete Topic", scale=1)
      with gr.Row():
        new_topic_input = gr.Textbox(label="New Topic Name", scale=2)
        create_topic_button = gr.Button("Create Topic", scale=1)

    with gr.Column():
      with gr.Row():
        subscription_dropdown = gr.Dropdown(
            label="Select a subscription", choices=[], interactive=True, scale=2
        )
        delete_subscription_button = gr.Button("Delete Subscription", scale=1)
      with gr.Row():
        new_subscription_input = gr.Textbox(
            label="New Subscription Name", scale=2
        )
        create_subscription_button = gr.Button("Create Subscription", scale=1)

      subscription_topic_display = gr.Dropdown(
          label="Subscribing to Topic:",
          choices=[],
          multiselect=False,
          interactive=False,
          visible=False,
      )

  loading_label = gr.Label(value="Loading...")

  # The main chat UI
  chat_output = gr.Chatbot(elem_id="chatbot", visible=False)
  chat_input = gr.MultimodalTextbox(
      interactive=True,
      file_count="multiple",
      placeholder="Enter message or upload file...",
      show_label=False,
      elem_id="chat_input",
      visible=False,
  )

  # Send message callback
  chat_input.submit(
      send_message,
      inputs=[
          chat_input,
          app_name_input,
          topic_dropdown,
          subscription_dropdown,
      ],
      outputs=[chat_output, chat_input, topic_dropdown, subscription_dropdown],
  )

  # Create Topic
  create_topic_button.click(
      fn=create_new_topic,
      inputs=[new_topic_input],
      outputs=[new_topic_input, topic_dropdown],
  )

  # Delete Topic
  delete_topic_button.click(
      fn=delete_topic,
      inputs=[topic_dropdown],
      outputs=[new_topic_input, topic_dropdown],
  )

  # Create Subscription
  def create_subscription_with_user_id_wrapper(
      topic_name_input, sub_name_input, request: gr.Request
  ):
    return create_subscription_with_user_id(
        topic_name_input, sub_name_input, request
    )

  create_subscription_button.click(
      fn=create_subscription_with_user_id_wrapper,
      inputs=[topic_dropdown, new_subscription_input],
      outputs=[
          new_subscription_input,
          subscription_dropdown,
          subscription_topic_display,
      ],
  )

  # Delete Subscription
  delete_subscription_button.click(
      fn=delete_subscription,
      inputs=[subscription_dropdown],
      outputs=[new_subscription_input, subscription_dropdown],
  )

  def update_subscription_topic_display(topic_name, subscription_name):
    """If the user picks a new single topic, display it in the 'Subscribing to Topic:' dropdown."""
    if isinstance(topic_name, str):
      topic_name = [topic_name]

    try:
      subscription_path = subscriber.subscription_path(
          project_id, subscription_name
      )
      sub = subscriber.get_subscription(
          request={"subscription": subscription_path}
      )
      expected_topic = sub.topic.split("/")[-1]
    except:
      expected_topic = "[Not found]"
    return gr.update(
        value=topic_name, label=f"Subscribing to Topic: {expected_topic}"
    )

  topic_dropdown.change(
      update_subscription_topic_display,
      inputs=[topic_dropdown, subscription_dropdown],
      outputs=[subscription_topic_display],
  )

  subscription_dropdown.change(
      update_subscription_topic_display,
      inputs=[topic_dropdown, subscription_dropdown],
      outputs=[subscription_topic_display],
  )

  # Show user from Google login
  demo.load(fn=show_user, inputs=None, outputs=[app_name_input])

  # Store global references on initialization
  def set_globals(chat, topic):
    """Initialize global references to the chat output & topic dropdown."""
    global active_chat_output, active_topic_dropdown
    active_chat_output = chat
    active_topic_dropdown = topic
    # Make chat UI visible
    return [gr.update(visible=True), gr.update(visible=True)]

  demo.load(
      fn=set_globals,
      inputs=[chat_output, topic_dropdown],
      outputs=[chat_output, chat_input],
  )

  # Load topics/subscriptions, create default subs (unique per topic), then show UI
  def load_all_and_show_ui(request: gr.Request):
    create_default_subscription_with_user_id(request)
    return [
        gr.update(choices=update_available_topics(), interactive=True),
        gr.update(choices=update_available_subscriptions(), interactive=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    ]

  demo.load(
      fn=load_all_and_show_ui,
      inputs=None,
      outputs=[
          topic_dropdown,
          subscription_dropdown,
          chat_output,
          chat_input,
          loading_label,
      ],
  )

gradio_app = gl.mount_gradio_app(app, demo, "/app")


if __name__ == "__main__":
  logging.info(f"Using project_id = {project_id}")
  script_name = os.path.basename(sys.argv[0]).replace(".py", "")
  uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=7860, reload=False)
