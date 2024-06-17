from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
import os
load_dotenv()
nvapi_key = os.environ.get("NVIDIA_API_KEY", "")


# import httpx, sys
llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)