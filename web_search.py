import os
import math
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")
URL = "https://www.googleapis.com/customsearch/v1"
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

    def _extract_domain(self, url: str) -> str:
        return url.replace("https://", "").replace("http://", "").split("/")[0]

    def search(self, query, max_results=3):
        if not self.cse_id or not self.api_key:
            raise ValueError("CSE_ID and API_KEY is required for web search.")
        
        if self.websites:
            site_query = " OR ".join([f"site:{self._extract_domain(url)}" for url in self.websites])
            query = f"{query} {site_query}"
            
        params = {"key": self.api_key, "cx": self.cse_id, "q": query, "num": max_results}
        response = requests.get(URL, params=params, timeout=5)
        response.raise_for_status()
        items = response.json().get("items", [])
        return [{"title": i["title"], "link": i["link"], "snippet": i.get("snippet")} for i in items]
