class AgentLogger:
    def __init__(self):
        self.role_log_mapping = {
            "route_image": "reader",
            "extract_ingredients": "reader",
            "rag_recipe_generator": "writer",
            "recipe_generator": "writer",
            "retrieve": "searcher",
            "grade_documents": "searcher",
            "decide_to_rag_generate": "searcher",
            "decide_not_to_rag_generate": "searcher",
            "check_hallucinations": "reviewer",
            "grade_generation_v_documents_and_question": "reviewer",
            "image_caption": "reader"
        }
        self.role_image_map = {
            "reader": "images/reader.png",
            "writer": "images/writer.png",
            "reviewer": "images/reviewer.png",
            "searcher": "images/searcher.png"
        }
        self.log_entries = []

    def clean_log_message(self, key, context):
        role = self.role_log_mapping.get(key, "unknown")
        if key == "extract_ingredients":
            if len(context["ingredients"]) > 0:
                return role, f'{role}: I found {", ".join(context["ingredients"])} in the food image'
            return role, f"{role}: I was not quite able to recognize ingredients from the image."
        if key == "rag_recipe_generator":
            return role, f"{role}: I finished writing the recipe with RAG, please review."
        if key == "recipe_generator":
            return role, f"{role}: I finished writing the recipe without RAG as instructed. Thank you for the feedback."
        if key == "route_image":
            image_type = context["image_type"]
            if image_type == "food_image":
                return role, f"{role}: The input image looks like a food image, let me see what ingredients I can see from it."
            return role, f"{role}: The input image is not a food image, but perhaps I could tell you what's in the image."
        if key == "retrieve":
            return role, f"{role}: I am checking the db for any relevant documents in the archive."
        if key == "decide_to_rag_generate":
            return role, f"{role}: The documents do appear to be recipes that may help with writing the recipe with RAG."
        if key == "decide_not_to_rag_generate":
            return role, f"{role}: I don't think the documents will help writer with recipe writing, I'd recommend not writng with RAG."
        if key == "grade_documents":
            return role, f"{role}: give me a moment to review the quality of recipes from archive"
        if key == "check_hallucinations":
             if context["check_hallucinations_result"] == "yes":
                return role, f"{role}: I can see that the Writer's work is grounded by the documents from archive"
             return role, f"{role}: The recipe is NOT grounded by the documents from archive, we should redo it."
        if key == "grade_generation_v_documents_and_question":
             if context["grade_generation_v_documents_and_question_grade"] == "yes":
                 return role, f"{role}: The Writer's work is a recipe for the food in the image."
             return role, f"{role}: I don't think the Writer's work is a recipe for the food in the image, we need to redo the work."
        if key == "image_caption":
            if 'generation' in context and context['generation'] and len(context['generation']) > 0:
                return role, f"{role}: {context['generation']}"
        return role, f"finished running {key}"

    def add_log_with_context(self, key, context):
        img_path = context.get("img_path", "")
        role, message = self.clean_log_message(key, context)
        if role in self.role_image_map:
            img_path = self.role_image_map[role]
        self.log_entries.append((img_path, message))

    def add_log_entry(self, img_path, message):
        self.log_entries.append((img_path, message))

    def get_log_entries(self):
        return self.log_entries

    def clear_log_entries(self):
        self.log_entries.clear()

# Instantiate the logger
logger = AgentLogger()