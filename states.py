from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    img_path: str
    question: str
    generation: str
    rag_generate: str
    documents: List[str]
    text: str
    ingredients: List[str]
    food_name: str


def route_image(state):
    """
    Route image to extract ingredients or image caption.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    is_food = detect_food_tool.invoke({"img_path": "/content/burger.png"})
    if is_food:
      return "extract_ingredients"
    return "image_caption"



def extract_ingredients(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---EXTRACT INGREDIENTS---")
    question = state["question"]
    documents = state["documents"]
    img_path = state["img_path"]
    text = state["text"]
    # img_path="/content/burger.png"
    # response = web_search_tool.invoke({"input":f' this is the image path: {img_path}'})
    # img_path = "/content/burger .png"
    # response = web_search_tool.invoke({"img_path":f'{img_path}'})
    # Web search
    response = extract_ingredients_tool.invoke({"img_path": img_path})
    print(response)
    # web_results = "\n".join([d["content"] for d in docs])
    # web_results = Document(page_content=web_results)
    # if documents is not None:
        # documents.append(web_results)
    # else:
        # documents = [web_results]
    return {"documents": [], "ingredients": [], "food_name": "", "text": "ingredients"}



def image_caption(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    # img_path="/content/burger.png"
    # response = web_search_tool.invoke({"input":f' this is the image path: {img_path}'})
    img_path = "/content/burger .png"
    return {"documents": [], "ingredients": [], "food_name": "", "question": question, "text": "caption"}


def recipe_generator(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---RECIPE GENERATOR SEARCH---")
    question = state["question"]
    documents = state["documents"]
    text = state["text"]
    print(text)
    # img_path="/content/burger.png"
    response = generate_recipe_tool.invoke({"input": text})
    print(response)
    # img_path = "/content/burger.png"
    return {"documents": [], "ingredients": [], "food_name": "", "question": question, "generation": response}