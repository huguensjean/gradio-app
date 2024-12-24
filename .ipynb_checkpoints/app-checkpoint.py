import gradio as gr

def greet():
   return "Hello, World!"

iface = gr.Interface(fn=greet, inputs=None, outputs="text")

if __name__ == "__main__":
   iface.launch(server_name="0.0.0.0", server_port=7860)
