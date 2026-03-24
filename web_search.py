import os
import math
import requests
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")
URL = os.getenv("URL")
VALID_LINKS = os.getenv("VALID_LINKS")


class WebSearch:
      
      def __init__(self):
        self.api_key = API_KEY
        self.cse_id = CSE_ID            
        self.websites = VALID_LINKS

        def search(query, max_results):
            if not self.cse_id or not self.api_key:
                raise ValueError("CSE_ID and API_KEY is required for web search.")
            
            params = {"key": self.api_key, "cse": self.cse_id, "q": query, "num": max_results}
            response = requests.get(URL, params = params)
            response.raise_for_status()
            items = response.json().get("items", [])
            return [{"title": i["title"], "link": i["link"]} for i in items]
        



        




