import os
import gradio as gr
from openai import Client
from dotenv import load_dotenv

load_dotenv()

#LLM parameters
BASEURL = os.getenv("ILSE_URL")
APIKEY = os.getenv("LITELLM_APIKEY")
MODEL = os.getenv("MODEL_LLAMA")

print(APIKEY)

#load LLM
client = Client(base_url=BASEURL, api_key=APIKEY)

# Function to get chatbot response
def chatbot(user_input):

    response = client.chat.completions.create(
        model=MODEL, 
        messages=[{"role": "user", "content": user_input}]
    )

    return response.choices[0].message.content

# Create a Gradio interface
iface = gr.Interface(
    fn=chatbot, 
    inputs="text", 
    outputs="text", 
    title="Pattent AI Chatbot",
    description="Ask me anything!"
)

# Launch app
iface.launch(share=True)
