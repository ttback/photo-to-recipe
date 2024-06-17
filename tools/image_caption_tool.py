import requests
from langchain.tools import BaseTool
from tools.utils import fetch_outputs, img2base64_string
from dotenv import load_dotenv
import os
load_dotenv()

nvapi_key = os.environ['NVIDIA_API_KEY']
class ImageCaptionTool(BaseTool):
    name = "Image captioner from phi-3"
    description = "Use this tool when given the path to an image that you would like to be described. " \
                  "It will return a simple caption describing the image."

    def _run(self, img_path):
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
              "content": f'what is in this image <img src="data:image/png;base64,{image_b64}" />'
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