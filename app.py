"""Gradio application for Gemini GenAI SDK."""

import base64
import enum
import importlib
import io
import logging
import mimetypes
import re
from typing import Any, Dict, List, Tuple

import cv2
# GCP utility functions (as in your original script)
from gcp_utils import authenticate
from gcp_utils import clean_resources_ui
from gcp_utils import get_project_id
from gcp_utils import get_project_number
from gcp_utils import get_region
# Google GenAI imports
from google import genai
from google.genai import types
import gradio as gr
import numpy as np
from PIL import Image
import pydantic


RawReferenceImage = types.RawReferenceImage
MaskReferenceImage = types.MaskReferenceImage
BaseModel = pydantic.BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = get_project_id()
PROJECT_NUMBER = get_project_number()
REGION = get_region()

# Instantiate GenAI Client with Vertex AI
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

# For referencing your original utilities
common_util = importlib.import_module(
    "vertex-ai-samples.community-content.vertex_model_garden.model_oss.notebook_util.common_util"
)

# The global variable to store the current image
# context for the current session.
current_file_context = None


def parse_bounding_boxes_pixels(
    text: str, width: int, height: int
) -> List[List[int]]:
  """Parse bounding boxes (normalized) from model response and convert to pixel coordinates."""
  pattern = r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]"
  matches = re.findall(pattern, text)
  bboxes = []
  for match in matches:
    y_min_norm, x_min_norm, y_max_norm, x_max_norm = map(float, match)
    # Convert from normalized [0,1] to pixel coordinates
    x_min = int(round(x_min_norm * width))
    x_max = int(round(x_max_norm * width))
    y_min = int(round(y_min_norm * height))
    y_max = int(round(y_max_norm * height))

    # Make sure coordinates are in the correct order
    if x_min > x_max:
      x_min, x_max = x_max, x_min
    if y_min > y_max:
      y_min, y_max = y_max, y_min

    bboxes.append([x_min, x_max, y_min, y_max])
  return bboxes


def plot_bounding_boxes_pixels(
    im: Image.Image, bboxes: List[List[int]]
) -> Image.Image:
  """Given a PIL image and bounding boxes in pixel coordinates, draw them with OpenCV."""
  image_np = np.array(im)
  image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

  for x_min, x_max, y_min, y_max in bboxes:
    # Double-check ordering
    if x_min > x_max:
      x_min, x_max = x_max, x_min
    if y_min > y_max:
      y_min, y_max = y_max, y_min

    cv2.rectangle(
        image_cv,
        (x_min, y_min),
        (x_max, y_max),
        color=(0, 0, 255),  # BGR red
        thickness=2,
    )

  image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
  return Image.fromarray(image_rgb)


################################################################################
# Example function for the "Function Calling" feature
################################################################################
def get_current_weather(location: str) -> str:
  """Example function that returns weather."""
  return f"The weather in {location} is always sunny in this demo!"


FUNCTION_CALL_TOOL = types.Tool(
    function_declarations=[{
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
            },
            "required": ["location"],
        },
    }]
)


################################################################################
# Example Pydantic Model for JSON schema responses
################################################################################
class CountryInfo(BaseModel):
  name: str
  population: int
  capital: str
  continent: str
  gdp: int
  official_language: str
  total_area_sq_mi: int


################################################################################
# Tasks enumeration
################################################################################
class Task(enum.Enum):
  """Tasks enumeration."""

  VQA = "VQA"
  CAPTION = "Image Captioning"
  OCR = "OCR"
  DETECT = "Object Detection"
  EMBED_TEXT = "Embed Text"
  COUNT_TOKENS = "Count Tokens"
  IMAGE_GEN = "Generate Image"
  IMAGE_UPSCALE = "Upscale Image"
  IMAGE_EDIT = "Edit Image"
  FILE_SUMMARY = "Summarize File (PDF)"
  AUDIO_QA = "Audio QnA"
  VIDEO_QA = "Video QnA"
  JSON_SCHEMA = "JSON Schema Demo"
  FUNCTION_CALL = "Function Calling"
  STREAMING = "Streaming Text Generation"


################################################################################
# Utility methods
################################################################################
def list_deployed_endpoints() -> List[str]:
  """Returns all valid endpoints or model identifiers that a user might want to use."""
  model_names = [
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-2.0-flash-exp",
      "gemini-2.0-thinking-exp-1219",
      "imagen-3.0-generate-001",  # for image generation example
      "imagen-3.0-capability-001",  # for image editing example
  ]
  return model_names


################################################################################
# Main Predict Handler
################################################################################
def predict_handler(
    vertex_endpoint_name: str,
    files: List[str],
    prompt: str,
    selected_task: str,
) -> str:
  """Route user input to the appropriate GenAI SDK method based on the selected task."""
  if not vertex_endpoint_name:
    raise gr.Error("Select a model first!")

  global current_file_context

  # If `files` is empty but we have a `current_file_context`,
  # fall back to that. Otherwise, just use the current `files`.
  files = files or current_file_context

  # Update the context for future usage
  current_file_context = files

  first_file_path = None
  file_mime = None
  file_bytes = None
  if files:
    first_file_path = files[0]
    file_mime, _ = mimetypes.guess_type(first_file_path)
    with open(first_file_path, "rb") as f:
      file_bytes = f.read()

  # ----------------------------------------------------------------------
  # DETECT logic: ask for bounding boxes, parse them, plot them, return them
  # ----------------------------------------------------------------------
  if selected_task == Task.DETECT.value:
    if not file_bytes:
      return "You must upload an image/audio/video for this task!"
    prompt += (
        " and return the bounding boxes of the objects in the image in the"
        " normalized format of [ [y_min, x_min, y_max, x_max], ...]."
    )
    contents = [
        prompt,
        types.Part.from_bytes(
            data=file_bytes,
            mime_type=file_mime or "application/octet-stream",
        ),
    ]
    response = client.models.generate_content(
        model=vertex_endpoint_name, contents=contents
    )
    pil_im = Image.open(io.BytesIO(file_bytes))
    bboxes = parse_bounding_boxes_pixels(
        response.text, pil_im.width, pil_im.height
    )
    if not bboxes:
      return response.text
    annotated_im = plot_bounding_boxes_pixels(pil_im, bboxes)
    annotated_im_html = _pil_image_to_html(annotated_im)
    return f"Bounding boxes found: {bboxes}\n\n{annotated_im_html}"

  # -- 1. TEXT GENERATION / VQA / CAPTION / OCR / AUDIO_QA / VIDEO_QA
  elif selected_task in {
      Task.VQA.value,
      Task.CAPTION.value,
      Task.OCR.value,
      Task.AUDIO_QA.value,
      Task.VIDEO_QA.value,
  }:
    if not file_bytes:
      return "You must upload an image/audio/video for this task!"
    contents = [
        prompt,
        types.Part.from_bytes(
            data=file_bytes,
            mime_type=file_mime or "application/octet-stream",
        ),
    ]
    response = client.models.generate_content(
        model=vertex_endpoint_name, contents=contents
    )
    return response.text

  # -- 2. EMBED TEXT --
  elif selected_task == Task.EMBED_TEXT.value:
    if not prompt.strip():
      return "Please enter some text to embed."
    embed_response = client.models.embed_content(
        model="text-embedding-004",
        contents=[prompt],
    )
    return f"Embeddings: {embed_response.embeddings}"

  # -- 3. COUNT TOKENS --
  elif selected_task == Task.COUNT_TOKENS.value:
    if not prompt.strip():
      return "Please enter some text to count tokens."
    tokens_response = client.models.count_tokens(
        model=vertex_endpoint_name, contents=prompt
    )
    return f"Token count: {tokens_response}"

  # -- 4. IMAGE GENERATION --
  elif selected_task == Task.IMAGE_GEN.value:
    if not prompt.strip():
      return "Please enter a prompt for image generation."
    img_response = client.models.generate_image(
        model=vertex_endpoint_name,
        prompt=prompt,
        config=types.GenerateImageConfig(
            negative_prompt="human",
            number_of_images=1,
            include_rai_reason=True,
            output_mime_type="image/jpeg",
        ),
    )
    if not img_response.generated_images:
      return "No images generated."
    pil_img = img_response.generated_images[0].image._pil_image
    return _pil_image_to_html(pil_img)

  # -- 5. IMAGE UPSCALING --
  elif selected_task == Task.IMAGE_UPSCALE.value:
    if not file_bytes:
      return "Please upload an image to upscale."
    upscale_response = client.models.upscale_image(
        model=vertex_endpoint_name,
        image=types.Image(image_bytes=file_bytes),
        config=types.UpscaleImageConfig(
            upscale_factor="x2",
        ),
    )
    if not upscale_response.generated_images:
      return "Upscale failed or not supported by this model."
    upscaled_pil = upscale_response.generated_images[0].image._pil_image
    return _pil_image_to_html(upscaled_pil)

  # -- 6. IMAGE EDITING --
  elif selected_task == Task.IMAGE_EDIT.value:
    if not file_bytes:
      return "Please upload an image to edit."

    base_img = RawReferenceImage(
        reference_id=1,
        reference_image=types.Image(image_bytes=file_bytes),
    )
    mask_img = MaskReferenceImage(
        reference_id=2,
        config=types.MaskReferenceConfig(
            mask_mode="MASK_MODE_BACKGROUND", mask_dilation=0
        ),
    )
    edit_response = client.models.edit_image(
        model=vertex_endpoint_name,
        prompt=prompt or "Sunlight and clear sky",
        reference_images=[base_img, mask_img],
        config=types.EditImageConfig(
            edit_mode="EDIT_MODE_INPAINT_INSERTION",
            number_of_images=1,
            negative_prompt="human",
            include_rai_reason=True,
            output_mime_type="image/jpeg",
        ),
    )
    if not edit_response.generated_images:
      return "Edit failed or not supported by this model."
    edited_pil = edit_response.generated_images[0].image._pil_image
    return _pil_image_to_html(edited_pil)

  # -- 7. FILE SUMMARY --
  elif selected_task == Task.FILE_SUMMARY.value:
    if not file_bytes or (file_mime != "application/pdf"):
      return "Please upload a PDF for summarization."
    contents = [
        types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
        prompt or "Please summarize the PDF.",
    ]
    response = client.models.generate_content(
        model=vertex_endpoint_name, contents=contents
    )
    return response.text

  # -- 8. JSON SCHEMA DEMO --
  elif selected_task == Task.JSON_SCHEMA.value:
    if not prompt.strip():
      prompt = "Give me information about the United States."
    response = client.models.generate_content(
        model=vertex_endpoint_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CountryInfo,
        ),
    )
    return response.text

  # -- 9. FUNCTION CALLING --
  elif selected_task == Task.FUNCTION_CALL.value:
    if not prompt.strip():
      prompt = "What is the weather in Boston?"
    response = client.models.generate_content(
        model=vertex_endpoint_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
        ),
    )
    return response.text

  # -- 10. STREAMING --
  elif selected_task == Task.STREAMING.value:
    if not prompt.strip():
      return "Please enter some text for streaming."
    streamed_text = []
    for chunk in client.models.generate_content_stream(
        model=vertex_endpoint_name, contents=prompt
    ):
      streamed_text.append(chunk.text)
    return "".join(streamed_text)

  else:
    return f"Unknown task: {selected_task}"


def _pil_image_to_html(img_pil: Image.Image) -> str:
  buffered = io.BytesIO()
  img_pil.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  html = f'<img src="data:image/png;base64,{img_str}"/>'
  return html


################################################################################
# Gradio Bot Handler
################################################################################
def bot_streaming(
    message,
    history,
    selected_task,
    vertex_endpoint_name=None,
):
  txt = message.get("text", "")
  files = message.get("files", [])
  logging.info("User message: %s, files: %s", txt, files)
  logging.info("history: %s", history)

  if vertex_endpoint_name:
    response = predict_handler(
        vertex_endpoint_name=vertex_endpoint_name,
        files=files,
        prompt=txt,
        selected_task=selected_task,
    )
  else:
    response = "No endpoints found. Please deploy/select a model first!"
  return response


################################################################################
# Initialization stub
################################################################################
def initialize_handler(
    create_bucket: bool,
    bucket_uri: str,
) -> Tuple[str, Dict[str, Any], gr.update]:
  return (
      "Initialization failed: Not implemented",
      None,
      gr.update(),
  )

################################################################################
# (NEW) We store the same examples from your gr.Examples in a Python list
################################################################################
EXAMPLE_LIST = [
    [
        {
            "text": "What is the woman doing?",
            "files": [
                "https://cdn.pixabay.com/photo/2023/08/08/09/21/couple-8176869_1280.jpg"
            ],
        },
        Task.VQA.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {
            "text": "Caption this image.",
            "files": [
                "https://cdn.pixabay.com/photo/2022/12/05/22/08/open-book-7637805_1280.jpg"
            ],
        },
        Task.CAPTION.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {
            "text": "OCR this image.",
            "files": [
                "https://cdn.pixabay.com/photo/2020/08/31/00/23/light-box-5531025_1280.jpg"
            ],
        },
        Task.OCR.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {
            "text": "Detect the birds.",
            "files": [
                "https://t3.ftcdn.net/jpg/06/85/16/06/360_F_685160679_UCvIfwz6weQVmmyhEpBdnN3QgWAbnNHv.jpg"
            ],
        },
        Task.DETECT.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "Please embed this sentence."},
        Task.EMBED_TEXT.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "Count tokens in this text."},
        Task.COUNT_TOKENS.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {
            "text": (
                "A beautiful painting of a panda riding a bike through a"
                " meadow."
            )
        },
        Task.IMAGE_GEN.value,
        "imagen-3.0-generate-001",
    ],
    [
        {
            "text": "Upscale this image please.",
            "files": [
                "https://t4.ftcdn.net/jpg/00/81/95/19/360_F_81951964_WW9pQsXZ4OwshXHEySZXHcuy7mEFNF9k.jpg"
            ],
        },
        Task.IMAGE_UPSCALE.value,
        "imagen-3.0-generate-001",
    ],
    [
        {
            "text": "Remove background and change it to a sunny beach.",
            "files": [
                "https://www.shutterstock.com/image-photo/group-pets-together-outdoors-summer-600nw-2130577739.jpg"
            ],
        },
        Task.IMAGE_EDIT.value,
        "imagen-3.0-capability-001",
    ],
    [
        {"text": "Summarize the attached PDF"},
        Task.FILE_SUMMARY.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "What is said in this audio?"},
        Task.AUDIO_QA.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "What is happening in this video?"},
        Task.VIDEO_QA.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "Give me information about Japan."},
        Task.JSON_SCHEMA.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "What is the weather in Boston?"},
        Task.FUNCTION_CALL.value,
        "gemini-2.0-flash-exp",
    ],
    [
        {"text": "Stream a short story about dragons."},
        Task.STREAMING.value,
        "gemini-2.0-flash-exp",
    ],
]


################################################################################
# (NEW) Helper functions to step forward/back through examples
################################################################################
def load_example_by_index(idx: int):
  """Load a single example by index, returns (msg, task, model)."""
  idx = idx % len(EXAMPLE_LIST)
  example = EXAMPLE_LIST[idx]
  message = example[0]  # dict with {'text':..., 'files': [...]} or none
  task = example[1]  # e.g. Task.VQA.value
  endpoint = example[2]  # e.g. "gemini-2.0-flash-exp"
  # Return them in a form Gradio can apply to the input widgets
  return idx, message, task, endpoint


def next_example(curr_idx):
  """Goes to the next example index."""
  return load_example_by_index(curr_idx + 1)


def prev_example(curr_idx):
  """Goes to the previous example index."""
  return load_example_by_index(curr_idx - 1)


################################################################################
# Main Gradio UI
################################################################################
with gr.Blocks(
    title="Model Garden PlaySpace for Gemini GenAI SDK",
    css="#chatbot {overflow:auto; height:500px;} footer {visibility: none}",
) as demo:
  gr.Markdown("""
        ## <span style="color:#4285F4;">G</span><span style="color:#EA4335;">o</span><span style="color:#FBBC05;">o</span><span style="color:#4285F4;">g</span><span style="color:#34A853;">l</span><span style="color:#EA4335;">e</span> Model Garden PlaySpace for Gemini GenAI SDK
        [Model Card](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/363) | [Notebook](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_jax_paligemma_deployment.ipynb) | [arXiv](https://arxiv.org/abs/2407.07726) | [Github](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/paligemma)
    """)

  logout_button = gr.Button("Logout", link="/logout")

  with gr.Row():
    with gr.Column(scale=3):
      with gr.Row(equal_height=True):
        app_state = gr.State({
            "MODEL_BUCKET": "",
            "SERVICE_ACCOUNT": "",
        })
        with gr.TabItem("Upload Media"):
          with gr.Accordion("Take a Photo", open=True):
            camera_image = gr.Image(
                type="filepath",
                interactive=True,
                elem_id="image_upload",
                label="Take a Photo",
                show_label=False,
                height=360,
            )
          with gr.Accordion(
              "Upload Multiple Files (Audio/Video/PDF)", open=False
          ):
            uploaded_files = gr.Files(
                type="filepath",
                label="Upload Files",
                show_label=False,
                file_count="multiple",
                interactive=True,
                elem_id="image_upload",
            )
          with gr.Row():
            endpoint_name = gr.Dropdown(
                scale=7,
                label="Select a model.",
                choices=list_deployed_endpoints(),
                interactive=True,
            )
            refresh_button = gr.Button(
                "Refresh Endpoints list",
                scale=1,
                variant="primary",
                min_width=10,
            )
          all_tasks = [
              Task.VQA.value,
              Task.CAPTION.value,
              Task.OCR.value,
              Task.DETECT.value,
              Task.EMBED_TEXT.value,
              Task.COUNT_TOKENS.value,
              Task.IMAGE_GEN.value,
              Task.IMAGE_UPSCALE.value,
              Task.IMAGE_EDIT.value,
              Task.FILE_SUMMARY.value,
              Task.AUDIO_QA.value,
              Task.VIDEO_QA.value,
              Task.JSON_SCHEMA.value,
              Task.FUNCTION_CALL.value,
              Task.STREAMING.value,
          ]
          selected_task = gr.Radio(
              choices=all_tasks,
              label="Select Task",
              value=Task.VQA.value,
              interactive=True,
          )
        with gr.TabItem("Model Settings", visible=True, interactive=True):
          with gr.Accordion("Open to Initialize Application", open=True):
            create_bucket_checkbox = gr.Checkbox(
                label="Create GCS Bucket", value=False
            )
            bucket_uri_input = gr.Textbox(
                label=(
                    "Bucket URI (e.g., gs://your-bucket-name). "
                    "Leave blank to use a default bucket."
                ),
                value="",
                interactive=False,
            )
            initialize_button = gr.Button(
                "Initialize", variant="primary", min_width=10
            )
            initialization_status = gr.Markdown(
                "**Initialization Status:** New model deployment takes "
                "about 15 minutes. Check progress at [Vertex Online "
                "Prediction](https://console.cloud.google.com/vertex-ai/online-prediction/endpoints)."
            )
            with gr.Row():
              choices = [
                  "paligemma-mix-224-float32",
              ]
              selected_model = gr.Dropdown(
                  scale=7,
                  label="Deploy a new model to Vertex",
                  choices=choices,
                  value=choices[0],
              )
              deploy_model_button = gr.Button(
                  "Deploy a new model",
                  scale=1,
                  variant="primary",
                  min_width=10,
              )
          with gr.Accordion("Clean Up Resources", open=False):
            with gr.Row():
              cleanup_project_id = gr.Textbox(
                  label="Project ID", value=PROJECT_ID, interactive=False
              )
              cleanup_region = gr.Textbox(
                  label="Region", value=REGION, interactive=False
              )
            with gr.Row():
              cleanup_endpoint_name = gr.Dropdown(
                  label="Select Endpoint to Delete",
                  choices=list_deployed_endpoints(),
                  interactive=True,
                  scale=7,
              )
              refresh_cleanup_endpoints_button = gr.Button(
                  "Refresh Endpoints List",
                  scale=1,
                  variant="primary",
                  min_width=10,
              )
            cleanup_delete_bucket = gr.Checkbox(label="Delete Bucket")
            cleanup_bucket_name = gr.Textbox(
                label="Bucket Name",
                placeholder="Enter bucket name if 'Delete Bucket' is checked",
            )
            cleanup_button = gr.Button("Clean Resources")
            cleanup_output = gr.Textbox(label="Cleanup Output")

            cleanup_button.click(
                fn=clean_resources_ui,
                inputs=[
                    cleanup_project_id,
                    cleanup_region,
                    cleanup_endpoint_name,
                    cleanup_delete_bucket,
                    cleanup_bucket_name,
                ],
                outputs=cleanup_output,
            )

            refresh_cleanup_endpoints_button.click(
                fn=lambda: gr.update(choices=list_deployed_endpoints()),
                outputs=cleanup_endpoint_name,
            )
          with gr.Accordion("Expand for additional help", open=False):
            with gr.TabItem("Help Section"):
              gr.Markdown("""
                                ### How to Use the GenAI Gradio Application
                                1. Select a task from the left sidebar.
                                2. Upload relevant files (images, pdf, audio, or video).
                                3. Enter your prompt.
                                4. Click "Submit" in the chatbox to see results.
                            """)

    with gr.Column(scale=7, elem_id="col"):
      chatbot = gr.Chatbot(
          elem_id="chatbot",
          render=False,
          bubble_full_width=True,
          min_height=680,
      )
      textbox = gr.MultimodalTextbox(
          elem_id="textbox",
          placeholder="Upload files and/or enter text prompt for the task.",
          label="Files + Text Prompt",
      )

      # The main ChatInterface
      chat = gr.ChatInterface(
          fn=bot_streaming,
          chatbot=chatbot,
          textbox=textbox,
          additional_inputs=[selected_task, endpoint_name],
          stop_btn="Stop",
          fill_height=True,
          multimodal=True,
      )

      with gr.Accordion("Click to try various examples", open=False):
        with gr.Row():
          prev_btn = gr.Button("Previous Example")
          next_btn = gr.Button("Next Example")
        gr.Examples(
            examples=EXAMPLE_LIST,  # Re-use the same EXAMPLE_LIST
            inputs=[
                chat.textbox,
                selected_task,
                endpoint_name,
            ],
            outputs=chat.chatbot,
            fn=bot_streaming,
            cache_examples=False,
            elem_id="examples",
        )

        # (NEW) Add "Previous Example" and "Next Example" Buttons
        example_index = gr.State(value=0)

        # Each button loads the example by index, updates the State, plus sets UI
        # to the correct text/files, task, and endpoint:
        def apply_example(idx, message, task, endpoint):
          # message is { 'text': '...', 'files': [...] } possibly
          # We return updates for (textbox, selected_task, endpoint_name)
          return (
              gr.update(value=message),
              gr.update(value=task),
              gr.update(value=endpoint),
              idx,  # store new index in state
          )

      prev_btn.click(
          fn=prev_example,
          inputs=[example_index],
          outputs=[example_index, textbox, selected_task, endpoint_name],
      ).then(
          fn=apply_example,
          inputs=[example_index, textbox, selected_task, endpoint_name],
          outputs=[textbox, selected_task, endpoint_name, example_index],
      )

      next_btn.click(
          fn=next_example,
          inputs=[example_index],
          outputs=[example_index, textbox, selected_task, endpoint_name],
      ).then(
          fn=apply_example,
          inputs=[example_index, textbox, selected_task, endpoint_name],
          outputs=[textbox, selected_task, endpoint_name, example_index],
      )

      # Keep the original .change() triggers to update the textbox
      uploaded_files.change(
          fn=lambda uploaded_files, existing_input: gr.update(
              value={
                  "text": existing_input.get("text", ""),
                  "files": uploaded_files,
              }
          ),
          inputs=[uploaded_files, chat.textbox],
          outputs=chat.textbox,
      )
      camera_image.change(
          fn=lambda captured_image, existing_input: gr.update(
              value={
                  "text": existing_input.get("text", ""),
                  "files": [captured_image],
              }
          ),
          inputs=[camera_image, chat.textbox],
          outputs=chat.textbox,
      )

  create_bucket_checkbox.change(
      fn=lambda create_bucket: gr.update(interactive=create_bucket),
      inputs=create_bucket_checkbox,
      outputs=bucket_uri_input,
  )
  initialize_button.click(
      initialize_handler,
      inputs=[create_bucket_checkbox, bucket_uri_input],
      outputs=[initialization_status, app_state, endpoint_name],
  )
  refresh_button.click(
      fn=lambda: gr.update(choices=list_deployed_endpoints()),
      outputs=endpoint_name,
  )

demo.queue()
demo.launch(
    share=True,
    inline=False,
    debug=True,
    show_error=True,
    server_port=7860,
    auth=authenticate,
)
