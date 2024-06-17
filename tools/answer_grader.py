from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import JsonOutputParser

model_id = "meta/llama3-70b-instruct"

llm = ChatNVIDIA(model=model_id, temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
    generated recipe is relevant to the food from image. Give a binary score 'yes' or 'no' to indicate whether the generated recipe
    is for the correct food name if food name is known. If food name is not known, answer on whether the generated recipe satisfy 
    user's question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    generated recipe
    {generation}
    \n ------- \n
    Here is the other context
    food name: {food_name}
    ingredients: {ingredients}
    user question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question", "food_name", "ingredients"],
)

answer_grader = prompt | llm | JsonOutputParser()