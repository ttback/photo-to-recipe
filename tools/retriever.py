### Index

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

urls = [
    "https://www.food.com/recipe/greek-style-turkey-burgers-13285",
    "https://www.food.com/recipe/kittencals-juicy-hamburger-burger-208583",
    # "https://www.food.com/recipe/california-rolls-japanese-244476",
    "https://www.food.com/recipe/the-perfect-burger-92021"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)
# from pprint import pprint
# pprint(doc_splits)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
)
retriever = vectorstore.as_retriever()