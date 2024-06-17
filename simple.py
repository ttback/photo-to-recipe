import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv()
nvapi_key = os.environ.get("NVIDIA_API_KEY", "")


import httpx, sys

import base64, io
from PIL import Image
import requests, json

def fetch_outputs(output):
    collect_streaming_outputs=[]
    for o in output:
        try:
            start = o.index('{')
            jsonString=o[start:]
            d = json.loads(jsonString)
            temp=d['choices'][0]['delta']['content']
            collect_streaming_outputs.append(temp)
        except:
            pass
    outputs=''.join(collect_streaming_outputs)
    return outputs.replace('\\','').replace('\'','')

def img2base64_string(img_path):
    image = Image.open(img_path)
    if image.width > 800 or image.height > 800:
        image.thumbnail((800, 800))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return image_base64



def fuyu(prompt,img_path):
    # invoke_url = "https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b"
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-mistral-7b"
    stream = True
    
    
    image_b64=img2base64_string(img_path)
    
    
    assert len(image_b64) < 200_000, \
      "To upload larger images, use the assets API (see docs)"

    headers = {
      "Authorization": f"Bearer {nvapi_key}",
      "Accept": "text/event-stream" if stream else "application/json"
    }
    
    payload = {
      "messages": [
        {
          "role": "user",
          "content": f'{prompt} <img src="data:image/png;base64,{image_b64}" />'
        }
      ],
      "max_tokens": 1024,
      "temperature": 0.20,
      "top_p": 0.70,
      "seed": 0,
      "stream": stream
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if stream:
        output=[]
        for line in response.iter_lines():
            if line:
                output.append(line.decode("utf-8"))
    else:
        output=response.json()
    out=fetch_outputs(output)
    return out

img_path="jordan.png"
prompt="describe the image"
out=fuyu(prompt,img_path)


from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", nvidia_api_key=nvapi_key, max_tokens=1024)
# llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1", nvidia_api_key=nvapi_key)
# llm = ChatNVIDIA(model="google/gemma-7b", nvidia_api_key=nvapi_key, max_tokens=1024)



import os
import io
# import IPython.display
from PIL import Image
import base64
import requests
import gradio as gr





#Set up Prerequisites for Image Captioning App User Interface
import os
import io
# import IPython.display
from PIL import Image
import base64
import requests
import gradio as gr

from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
# import torch
#
import os
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    
def fetch_outputs(output):
    collect_streaming_outputs=[]
    for o in output:
        try:
            start = o.index('{')
            jsonString=o[start:]
            d = json.loads(jsonString)
            temp=d['choices'][0]['delta']['content']
            collect_streaming_outputs.append(temp)
        except:
            pass
    outputs=''.join(collect_streaming_outputs)
    return outputs.replace('\\','').replace('\'','')

def img2base64_string(img_path):
    image = Image.open(img_path)
    if image.width > 800 or image.height > 800:
        image.thumbnail((800, 800))
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return image_base64


# class ImageCaptionTool(BaseTool):
#     name = "Image captioner from Fuyu"
#     description = "Use this tool when given the path to an image that you would like to be described. " \
#                   "It will return a simple caption describing the image."

#     def _run(self, img_path):
#         invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-mistral-7b"
#         stream = True


#         image_b64=img2base64_string(img_path)


#         assert len(image_b64) < 200_000, \
#           "To upload larger images, use the assets API (see docs)"
#         headers = {
#           "Authorization": f"Bearer {nvapi_key}",
#           "Accept": "text/event-stream" if stream else "application/json"
#         }

#         payload = {
#           "messages": [
#             {
#               "role": "user",
#               "content": f'Describe the image of food, its ingredients, and dish name, output in json for each field <img src="data:image/png;base64,{image_b64}" />'
#             }
#           ],
#           "max_tokens": 1024,
#           "temperature": 0.20,
#           "top_p": 0.70,
#           "seed": 0,
#           "stream": stream
#         }

#         response = requests.post(invoke_url, headers=headers, json=payload)

#         if stream:
#             output=[]
#             for line in response.iter_lines():
#                 if line:
#                     output.append(line.decode("utf-8"))
#         else:
#             output=response.json()
#         out=fetch_outputs(output)
#         return out

#     def _arun(self, query: str):
#         raise NotImplementedError("This tool does not support async")


class ImageCaptionTool(BaseTool):
    name = "Image captioner from Fuyu"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
        # invoke_url = "https://integrate.api.nvidia.com/v1/mistralai/mixtral-8x22b-instruct-v0.1"
        # invoke_url = "https://ai.api.nvidia.com/v1/vlm/community/llava16-mistral-7b"
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct"
        stream = True


        image_b64=img2base64_string(img_path)


        assert len(image_b64) < 200_000, \
          "To upload larger images, use the assets API (see docs)"
        headers = {
          "Authorization": f"Bearer {nvapi_key}",
          "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
          "messages": [
            {
              "role": "user",
              "content": f'Describe the image of food, its ingredients, and dish name, output in json for each field <img src="data:image/png;base64,{image_b64}" />'
            }
          ],
          "max_tokens": 1024,
          "temperature": 0.20,
          "top_p": 0.70,
          "seed": 0,
          "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            output=[]
            for line in response.iter_lines():
                if line:
                    output.append(line.decode("utf-8"))
        else:
            output=response.json()
        out=fetch_outputs(output)
        return out

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")


class TabularPlotTool(BaseTool):
    name = "Tabular Plot reasoning tool"
    description = "Use this tool when given the path to an image that contain bar, pie chart objects. " \
                  "It will extract and return the tabular data "


    def _run(self, img_path):
        invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
        stream = True

        image_b64=img2base64_string(img_path)

        assert len(image_b64) < 180_000, \
          "To upload larger images, use the assets API (see docs)"

        headers = {
          "Authorization": f"Bearer {nvapi_key}",
          "Accept": "text/event-stream" if stream else "application/json"
        }

        payload = {
          "messages": [
            {
              "role": "user",
              "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
          ],
          "max_tokens": 1024,
          "temperature": 0.20,
          "top_p": 0.20,
          "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            output=[]
            for line in response.iter_lines():
                if line:
                    temp=line.decode("utf-8")
                    output.append(temp)
                    #print(temp)
        else:
            output=response.json()
        outputs=fetch_outputs(output)
        return outputs
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")



#initialize the gent
tools = [ImageCaptionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)


agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    handle_parsing_errors=True,
    early_stopping_method='generate'
)



def my_agent(img_path):
    # response = agent.invoke({"input":f'this is the image path: {img_path}'})
    # response = agent.invoke({"input":f'Compose a recipe using the ingredients in this image: {img_path}'})
    response = agent.invoke({"input":f'Compose a recipe with quuantity of ingredients and detailed step-by-step instruction based on the description and ingredients in this image: {img_path}'})

    return response['output']


import gradio as gr
ImageCaptionApp = gr.Interface(fn=my_agent,
                    inputs=[gr.Image(label="Upload image", type="filepath")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with langchain agent",
                    description="combine langchain agent using tools for image reasoning",
                    allow_flagging="never")

ImageCaptionApp.launch(share=False)