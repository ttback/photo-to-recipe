from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import JsonOutputParser

model_id = "meta/llama3-70b-instruct"
llm = ChatNVIDIA(model=model_id, temperature=0)

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    the generated recipe is similar to a set of other recipes found. Give a binary 'yes' or 'no' score to indicate
    whether the generated recipe is similar to other recipes found. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the other recipes found:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()