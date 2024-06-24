from langchain.schema import Document
import json
from tools.image_caption_tool import ImageCaptionTool
from tools.extract_ingredients_tool import ExtractIngredientsTool
from tools.detect_food_tool import DetectFoodTool
from tools.generate_recipe_tool import GenerateRecipeTool
from tools.retriever import retriever
from tools.retrieval_grader import retrieval_grader
from tools.rag_generate import rag_chain
from agent_logger import logger

# Initialize tools
extract_ingredients_tool = ExtractIngredientsTool()
image_caption_tool = ImageCaptionTool()
detect_food_tool = DetectFoodTool()
generate_recipe_tool = GenerateRecipeTool()

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    logger.add_log_with_context("retrieve", {"img_path": state.get("img_path", ""), "question": question})
    return {"documents": documents, "question": question}


def rag_generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---RAG GENERATE---")
    question = state["question"]
    documents = state["documents"]
    ingredients = state.get("ingredients", [])
    food_name = state.get("food_name", "")
    print(f"ingredients: {ingredients}")
    print(f"food_name: {food_name}")
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question, "ingredients": ",".join(ingredients), "food_name": food_name})
    print("---COMPLETE RAG GENERATE---")
    logger.add_log_with_context("rag_recipe_generator", {"img_path": state.get("img_path", ""), "documents": documents, "question": question, "generation": generation})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    rag_generate = "Yes"
    num_of_docs = len(documents)
    num_of_irrelevant_docs = 0
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we don't want to use rag_generate
            num_of_irrelevant_docs +=1
            continue
    if num_of_docs == 0:
      rag_generate = "No"
    if num_of_irrelevant_docs > (num_of_docs / 2):
      rag_generate = "No"
    print(f"rag_generate: {rag_generate}")
    logger.add_log_with_context("grade_documents", {"img_path": state.get("img_path", ""), "documents": filtered_docs, "question": question, "rag_generate": rag_generate})
    return {"documents": filtered_docs, "question": question, "rag_generate": rag_generate}


def web_search(state):
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
    response = web_search_tool.invoke({"img_path":f'{img_path}'})
    # Web search
    docs = web_search_tool.invoke({"input": question})
    # web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    logger.add_log_with_context("web_search", {"img_path": state.get("img_path", ""), "documents": documents, "question": question})
    return {"documents": documents, "question": question}




def extract_ingredients(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---EXTRACT INGREDIENTS---")
    # question = state["question"]
    # documents = state["documents"]
    img_path = state["img_path"]
    text = state["text"]
    # img_path="/content/burger.png"
    # response = web_search_tool.invoke({"input":f' this is the image path: {img_path}'})
    # img_path = "/content/burger .png"
    # response = web_search_tool.invoke({"img_path":f'{img_path}'})
    # Web search
    response = extract_ingredients_tool.invoke({"img_path": img_path})
    response = response.replace('```json', "").replace("```", "")
    # web_results = "\n".join([d["content"] for d in docs])
    # web_results = Document(page_content=web_results)
    # if documents is not None:
        # documents.append(web_results)
    # else:
        # documents = [web_results]
    # Check if the response is a JSON object
    try:
        response_dict = json.loads(response)
        description = response_dict.get("description", "")
        ingredients = response_dict.get("ingredients", [])
        dish_name = response_dict.get("dish_name", "")
        
        # Convert JSON to regular text
        text = f"Description: {description}\nIngredients: {', '.join(ingredients)}\nDish Name: {dish_name}"
        print(text)
        logger.add_log_with_context("extract_ingredients", {"img_path": img_path, "ingredients": ingredients, "description": description, "dish_name": dish_name})
        return {"documents": [], "ingredients": ingredients, "food_name": dish_name, "text": text}
    except json.JSONDecodeError:
        # Handle the case where the response is not a JSON object
        print(f"Non-json response:{response}")
        return {"documents": [], "ingredients": [], "food_name": "", "text": response}



def image_caption(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---IMAGE CAPTION---")
    question = state["question"]
    documents = state["documents"]
    img_path = state["img_path"]
    # img_path="/content/burger.png"
    response = image_caption_tool.invoke({"img_path":img_path})
    logger.add_log_with_context("image_caption", {"documents": [], "ingredients": [], "food_name": "", "question": question, "text": response, "generation": response})
    # img_path = "/content/burger .png"
    return {"documents": [], "ingredients": [], "food_name": "", "question": question, "text": response, "generation": response}


def recipe_generator(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---RECIPE GENERATOR WITHOUT RAG---")
    question = state["question"]
    documents = state["documents"]
    text = state["text"]
    print(text)
    # img_path="/content/burger.png"
    response = generate_recipe_tool.invoke({"input": text})
    print(response)
    # img_path = "/content/burger.png"
    logger.add_log_with_context("recipe_generator", {"img_path": state.get("img_path", ""), "question": question, "text": text, "generation": response})
    return {"documents": [], "ingredients": [], "food_name": "", "question": question, "generation": response}