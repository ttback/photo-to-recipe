from langchain.tools import BaseTool
from openai import OpenAI
import os


class GenerateRecipeTool(BaseTool):
    name = "Simple Recipe Generate tool"
    description = "Use this tool to generate recipe"

    def _run(self, input):
        stream = True
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"]
        )

        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[{"role": "user", "content": f"Compose a recipe with instructions and quantity of ingredients mentioned in the following text: {input}"}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=stream
        )

        if stream:
            output = []
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    output.append(chunk.choices[0].delta.content)
        else:
            output = response.json()
        out = "\n".join(output)
        return out

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
        