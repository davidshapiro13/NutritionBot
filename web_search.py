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
from prompts import main_system_prompt
from rag_pipeline import RAGPipeline
from location_service import LocationService
from AI import AI

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
