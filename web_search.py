import os
import math
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# import faiss
# from sentence_transformers import SentenceTransformer
# import pypdf
# from docx import Document as DocxDocument
from llmproxy import LLMProxy

from user_memory import UserMemory
from prompts import (
    main_system_prompt,
    button_creator_prompt,
    guided_transition_prompt,
    intent_classifier_prompt,
    WELCOME_BUTTONS,
    WELCOME_MESSAGE,
    FOOD_SAFETY_BUTTONS,
    NUTRITION_BUTTONS,
    STORE_TYPE_BUTTONS,
    WIC_INFO_BUTTONS,
    LOCATION_PROMPT,
)
from rag_pipeline import RAGPipeline
from location_service import LocationService
from AI import AI

from wa_service_sdk import Button
from user_memory import UserMemory

import ast
import re

VALID_LINKS = {
"https://www.nutrition.gov/", 
"https://www.cancer.gov/", 
"https://nutritionsource.hsph.harvard.edu/", 
"https://www.cspi.org/", 
"https://snaped.fns.usda.gov/resources/nutrition-education-materials/meal-planning-shopping-and-budgeting", 
"https://www.mayoclinic.org/symptom-checker/select-symptom/itt-20009075", 
"https://medlineplus.gov/foodandnutrition.html", 
"https://www.foodsafety.gov/"
}



class WebSearch:
      
      def __init__(self):
        self.api_key = API_KEY
        self.cse_id = CSE_ID            
        self.websites = VALID_LINKS

        def search(query, max_results):
            if not self.cse_id:
                raise ValueError("CSE_ID is required for Google search.")
            
            params = {"key": self.api_key, "cse": self.cse_id, "q": query, "num": max_results}
            response = requests.get(URL, params = params)
            response.raise_for_status()
            items = response.json().get("items", [])
            return [{"title": i["title"], "link": i["link"], "snippet": i["snippet"]} for i in items]
        



        




