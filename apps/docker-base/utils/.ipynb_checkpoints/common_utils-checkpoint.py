"""Common utils for pubsub."""

import json
import logging
import os
from typing import Callable, Optional

from google.api_core import exceptions
from google.api_core.future import Future
from google.auth.credentials import Credentials
from google.cloud import pubsub_v1

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration using environment variables
project_id = os.environ.get("PROJECT_ID", "")
topic_name = os.environ.get("TOPIC_NAME", "")
subscription_name = os.environ.get("SUBSCRIPTION_NAME", "")


def create_publisher(
    credentials: Optional[Credentials] = None,
) -> pubsub_v1.PublisherClient:
  """Creates and returns a Pub/Sub publisher client.

  Args:
      credentials (google.auth.credentials.Credentials, optional): Google Cloud
        credentials to use. Defaults to None (Application Default Credentials).

  Returns:
      pubsub_v1.PublisherClient: The Pub/Sub publisher client.
  """
  return pubsub_v1.PublisherClient(credentials=credentials)


def create_subscriber(
    credentials: Optional[Credentials] = None,
) -> pubsub_v1.SubscriberClient:
  """Creates and returns a Pub/Sub subscriber client.

  Args:
      credentials (google.auth.credentials.Credentials, optional): Google Cloud
        credentials to use. Defaults to None (Application Default Credentials).

  Returns:
      pubsub_v1.SubscriberClient: The Pub/Sub subscriber client.
  """
  return pubsub_v1.SubscriberClient(credentials=credentials)


def publish_message(
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    message: dict,
    enable_batching: bool = False,
) -> bool:
  """Publishes a message to a Pub/Sub topic.

  Args:
      publisher (pubsub_v1.PublisherClient): The Pub/Sub publisher client.
      topic_path (str): The full topic path (e.g.,
        "projects/my-project/topics/my-topic").
      message (dict): The message data (Python dictionary).
      enable_batching (bool, optional): Enables batching for publishing.
        Defaults to False.

  Returns:
      bool: True if the message was published successfully, False otherwise.
  """
  message_data = json.dumps(message).encode("utf-8")
  try:
    if enable_batching:
      future = publisher.publish(topic_path, message_data)
      future.add_done_callback(_publish_callback)
      logging.info(
          f"Message published asynchronously (batched) to {topic_path}"
      )
      return True
    else:
      future = publisher.publish(topic_path, message_data)
      future.result()  # Ensures sync publishing.
      logging.info(f"Message published successfully to {topic_path}")
      return True
  except exceptions.GoogleAPIError as e:
    logging.error(f"Error publishing message to {topic_path}: {e}")
    return False


def _publish_callback(future: Future) -> None:
  """Callback function for asynchronous publishing."""
  try:
    future.result()
    logging.debug(f"Message was successfully published in background.")
  except Exception as e:
    logging.error(f"Error publishing message in the background: {e}")


def pull_messages(
    subscriber: pubsub_v1.SubscriberClient,
    subscription_path: str,
    callback: Callable[[pubsub_v1.types.PubsubMessage], None],
    max_messages: int = 1000,
    flow_control: Optional[pubsub_v1.types.FlowControl] = None,
) -> Future:
  """Starts a subscription to a Pub/Sub subscription using a callback.

  Args:
      subscriber (pubsub_v1.SubscriberClient): The Pub/Sub subscriber client.
      subscription_path (str): The full subscription path.
      callback (callable): The callback function to process messages. It should
        accept a single argument of type pubsub_v1.types.PubsubMessage.
      max_messages (int, optional): The max number of messages to receive at a
        time. Defaults to 1000.
      flow_control (pubsub_v1.types.FlowControl, optional): Control the rate of
        messages being received. Defaults to None.

  Returns:
      google.api_core.future.Future: A future object that can be used to manage
      the subscription.
  """
  if flow_control is None:
    flow_control = pubsub_v1.types.FlowControl(max_messages=max_messages)

  streaming_pull_future = subscriber.subscribe(
      subscription_path, callback=callback, flow_control=flow_control
  )

  logging.info(
      f"Pulling messages from {subscription_path} with max_messages set to"
      f" {max_messages}"
  )
  return streaming_pull_future


def create_topic(
    publisher: pubsub_v1.PublisherClient, project_id: str, topic_name: str
) -> str:
  """Creates a Pub/Sub topic if it does not already exist.

  Args:
      publisher (pubsub_v1.PublisherClient): The Pub/Sub publisher client.
      project_id (str): The Google Cloud project ID.
      topic_name (str): The name of the topic to create.

  Returns:
     str: The topic path of the created topic or an existing topic.
  """
  topic_path = publisher.topic_path(project_id, topic_name)
  try:
    publisher.get_topic(request={"topic": topic_path})
    logging.info(f"Topic {topic_path} already exists.")
  except exceptions.NotFound:
    publisher.create_topic(request={"name": topic_path})
    logging.info(f"Topic {topic_path} created.")

  return topic_path


def create_subscription(
    subscriber: pubsub_v1.SubscriberClient,
    project_id: str,
    topic_name: str,
    subscription_name: str,
) -> str:
  """Creates a Pub/Sub subscription to a topic if it doesn't already exist.

  Args:
      subscriber (pubsub_v1.SubscriberClient): The Pub/Sub subscriber client.
      project_id (str): The Google Cloud project ID.
      topic_name (str): The name of the topic to subscribe to.
      subscription_name (str): The name of the subscription to create.

  Returns:
      str: The subscription path of the created subscription or an existing
      subscription.
  """
  topic_path = subscriber.topic_path(project_id, topic_name)
  subscription_path = subscriber.subscription_path(
      project_id, subscription_name
  )

  try:
    subscriber.get_subscription(request={"subscription": subscription_path})
    logging.info(f"Subscription {subscription_path} already exists.")
  except exceptions.NotFound:
    subscriber.create_subscription(
        request={"name": subscription_path, "topic": topic_path}
    )
    logging.info(
        f"Subscription {subscription_path} created for topic {topic_path}."
    )
  return subscription_path
