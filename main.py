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
from agent_logger import logger
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
    print("---ROUTE IMAGE---")
    
    question = state["question"]
    img_path = state["img_path"]
    if not img_path:
        return ""
    is_food = detect_food_tool.invoke({"img_path": img_path})
    if is_food:
        logger.add_log_with_context("route_image", {"img_path": state["img_path"], "image_type": "food_image"})
        return "food_image"
    logger.add_log_with_context("route_image", {"img_path": state["img_path"], "image_type": "image"})
    return "image"

def decide_to_rag_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    rag_generate = state["rag_generate"]
    filtered_documents = state["documents"]

    if rag_generate == "Yes":
        print("---DECISION: MOST DOCUMENTS ARE RELEVANT TO QUESTION ---")
        logger.add_log_with_context("decide_to_rag_generate", {"img_path": state["img_path"], "message": "---DECISION: MOST DOCUMENTS ARE RELEVANT TO QUESTION ---"})
        return "rag"
    else:
        print("---DECISION: GENERATE WITHOUT RAG---")
        logger.add_log_with_context("decide_not_to_rag_generate", {"img_path": state["img_path"], "message": "---DECISION: GENERATE WITHOUT RAG---"})
        return "generate"

def grade_generation_v_documents_and_question(state):
    print("---CHECK HALLUCINATIONS---")
 
    print("---LOG ENTRIES---")
    print(logger.get_log_entries())
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    ingredients = state.get("ingredients", [])
    food_name = state.get("food_name", "")
    print(f"Generation: {generation}")

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]
    logger.add_log_with_context("check_hallucinations", {"img_path": state["img_path"], "check_hallucinations_result": grade})
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        print("---GRADE GENERATION vs QUESTION---")
        
        score = answer_grader.invoke({"question": question, "generation": generation, "ingredients": ','.join(ingredients), "food_name": food_name})
        grade = score["score"]
        logger.add_log_with_context("grade_generation_v_documents_and_question", {"img_path": state["img_path"], "grade_generation_v_documents_and_question_grade": grade})
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
        "": END
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

# def recipe_agent(img_path):
#     # response = agent.invoke({"input":f'this is the image path: {img_path}'})
#     # response = agent.invoke({"img_path": "/mnt/c/Users/Ti3/Dev/nv-image-to-table/images/burger.png", "input":f'generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned): {img_path}'})
#     app = workflow.compile()
#     inputs = {"img_path": img_path, "question": "generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned)"}

#     for output in app.stream(inputs):
#       for key, value in output.items():
#           pprint(f"Finished running: {key}:")
#     return value["generation"]


import gradio as gr
# ImageToRecipeApp = gr.Interface(fn=recipe_agent,
#                     inputs=[gr.Image(label="Upload image", type="filepath")],
#                     outputs=[gr.Textbox(label="Recipe")],
#                     title="Photo to Recipe Generation with Generative AI Agents",
#                     description="Combine multiple langchain agents and tools for photo-based recipe generation",
#                     allow_flagging="never",
#                     theme=gr.themes.Soft())
# ImageToRecipeApp.launch(server_name="0.0.0.0", server_port=7860, share=True)

def recipe_agent(img_path):
    logger.clear_log_entries()  # Clear previous log entries
    app = workflow.compile()
    inputs = {"img_path": img_path, "question": "generate recipe with quantity of ingredients and step-by-step instructions (must include but not limited to all the ingredients mentioned)"}

    # Add a log entry to see if it shows up
    print("Starting recipe generation process")
    logger.add_log_entry(img_path, "Starting multi-agent meeting to work on the input image...")
    yield "", generate_log_html(logger.get_log_entries())  # Yield an empty recipe and the current logs
    value = None  # Initialize value to avoid UnboundLocalError
    if inputs:
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key} - {value}")
                yield "", generate_log_html(logger.get_log_entries())  # Yield an empty recipe and the current logs
        if value:
            yield value["generation"], generate_log_html(logger.get_log_entries())  # Yield the final recipe and the logs
        else:
            yield "", generate_log_html(logger.get_log_entries())

def format_message(message):
    parts = message.split(":", 1)
    if len(parts) == 2:
        return f'<strong style="text-transform: capitalize;">{parts[0]}</strong>: {parts[1]}'
    return message

def generate_log_html(log_entries):
    html_content = ""

    for img_path, message in log_entries:
        print(f"img_path: {img_path}")
        if img_path:
            if img_path.startswith("/tmp/gradio"):
                # Handle temporary uploaded images
                html_content += f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                html_content += f'<img src="{img_path}" alt="Image" style="width: 100px; height: 100px; margin-right: 10px;">'
                html_content += f'<span>{message}</span>'
                html_content += f'</div>'
            else:
# Handle preset assets from the images directory
                html_content += f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                html_content += f'<img src="{img_path}" alt="Image" style="width: 100px; height: 100px; margin-right: 10px;">'
                html_content += f'<span>{format_message(message)}</span>'
                html_content += f'</div>'
    return html_content

# Gradio components
image_input = gr.Image(label="Upload image", type="filepath")
recipe_output = gr.Markdown(label="Recipe")
log_viewer = gr.HTML()

# Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column():
            image_input.render()
            recipe_output.render()
        with gr.Column():
            with gr.Accordion("AI Team Meeting Transcript", open=True):
                log_viewer.render()

    image_input.change(fn=recipe_agent, inputs=image_input, outputs=[recipe_output, log_viewer])


# demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import gradio as gr
from threading import Thread

app = FastAPI()

# Serve static files from the '/tmp' directory
app.mount("/tmp", StaticFiles(directory="/tmp"), name="tmp")

# Serve static files from the 'images' directory
app.mount("/images", StaticFiles(directory="images"), name="images")

gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)