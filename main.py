import os
from pprint import pprint
from tools.image_caption_tool import ImageCaptionTool
from tools.extract_ingredients_tool import ExtractIngredientsTool
from tools.detect_food_tool import DetectFoodTool
from tools.generate_recipe_tool import GenerateRecipeTool
from tools.hallucination_grader import hallucination_grader
from tools.answer_grader import answer_grader
from states import GraphState
from nodes import retrieve, rag_generate, grade_documents, web_search, recipe_generator, image_caption, extract_ingredients
from langgraph.graph import END, StateGraph

# Initialize tools
extract_ingredients_tool = ExtractIngredientsTool()
image_caption_tool = ImageCaptionTool()
detect_food_tool = DetectFoodTool()
generate_recipe_tool = GenerateRecipeTool()

# Define workflow
workflow = StateGraph(GraphState)
# workflow.add_node("recipe_generator", generate_recipe_tool)
workflow.add_node("image_caption", image_caption)
workflow.add_node("extract_ingredients", extract_ingredients)
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("rag_recipe_generator", rag_generate)
workflow.add_node("recipe_generator", recipe_generator)
# workflow.add_node("web_search", web_search)

### Conditional edge


def route_image(state):
    """
    Route image to extract ingredients or image caption.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE IMAGE---")
    question = state["question"]
    img_path = state["img_path"]
    # print(question)
    is_food = detect_food_tool.invoke({"img_path": img_path})
    if is_food:
      return "food_image"
    return "image"


def decide_to_rag_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    rag_generate = state["rag_generate"]
    filtered_documents = state["documents"]

    if rag_generate == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: MOST DOCUMENTS ARE RELEVANT TO QUESTION ---"
        )
        return "rag"
    else:
        # More than half of the documents are not relevant, so generate answer
        print("---DECISION: GENERATE WITHOUT RAG---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    ingredients = state.get("ingredients", [])
    food_name = state.get("food_name", "")
    print(f"Generation: {generation}")

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation, "ingredients": ','.join(ingredients), "food_name": food_name})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


# Build graph
workflow.set_conditional_entry_point(
    route_image,
    {
        "food_image": "extract_ingredients",
        "image": "image_caption",
    },
)
# workflow.add_edge("extract_ingredients", "recipe_generator")
workflow.add_edge("extract_ingredients", "retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_rag_generate,
    {
        "rag": "rag_recipe_generator",
        "generate": "recipe_generator",
    },
)
# workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "rag_recipe_generator",
    grade_generation_v_documents_and_question,
    {
        "not supported": "recipe_generator",
        "useful": END,
        "not useful": "recipe_generator",
    },
)

# Compile workflow
app = workflow.compile()

# Test


# inputs = {"img_path": "/app/images/burger.png", "question": "generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned)"}


# img_path = '/app/images/cali-roll.png'
# img_path = '/app/images/burger.png'
# inputs = {"img_path": img_path, "question": "generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned)"}

# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(value)

def recipe_agent(img_path):
    # response = agent.invoke({"input":f'this is the image path: {img_path}'})
    # response = agent.invoke({"img_path": "/mnt/c/Users/Ti3/Dev/nv-image-to-table/images/burger.png", "input":f'generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned): {img_path}'})
    app = workflow.compile()
    inputs = {"img_path": img_path, "question": "generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned)"}

    for output in app.stream(inputs):
      for key, value in output.items():
          pprint(f"Finished running: {key}:")
    return value["generation"]


import gradio as gr
ImageToRecipeApp = gr.Interface(fn=recipe_agent,
                    inputs=[gr.Image(label="Upload image", type="filepath")],
                    outputs=[gr.Textbox(label="Recipe")],
                    title="Image to Recipe Generation with Generative AI Agents",
                    description="Combine multiple langchain agents and tools for image-based recipe generation",
                    allow_flagging="never",
                    theme=gr.themes.Soft())
ImageToRecipeApp.launch(server_name="0.0.0.0", server_port=7860, share=True)