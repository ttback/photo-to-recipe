from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
model_id = "meta/llama3-70b-instruct"
# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for receipt generation tasks.
    Use the following pieces of retrieved context of recipes to generate a recipe based on given ingredients.
    If you are unable to becauese the context and ingredients are not related to the food name, just say that you don't know.
    Recipe should have 3 sections, food name, ingredient list with quantity and instructions <|eot_id|><|start_header_id|>user<|end_header_id|>
    Food name(could be empty): {food_name}
    Context: {context}
    Ingredients: {ingredients}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document", "food_name", "ingredients"],
)

llm = ChatNVIDIA(model=model_id, temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()