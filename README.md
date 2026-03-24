# NutritionBot

A WhatsApp-based nutrition and food safety assistant powered by a RAG (Retrieval Augmented Generation) pipeline, per-user memory, and Google Places integration for finding nearby WIC-authorized stores.

---

## Features

- Answer questions about human nutrition, diet, and food safety
- WIC-approved food guidance based on the Massachusetts WIC Food Guide
- Food storage time recommendations (refrigerator & freezer)
- Per-user memory that remembers dietary restrictions, health goals, and allergies
- Nearby WIC store finder via Google Places API
- Topic guardrails that decline out-of-scope questions

---

## Project Structure

```
NutritionBot/
├── Main.py               # Starts the web server and ngrok tunnel
├── Nutrition_Bot.py      # WhatsApp webhook event handler
├── AI.py                 # LLMProxy wrapper (used when RAG is disabled)
├── rag_pipeline.py       # Public knowledge base retrieval + answer generation
├── user_memory.py        # Per-user profile memory (extract, store, retrieve)
├── location_service.py   # Google Places API — find nearby WIC stores
├── prompts.py            # Unified system prompt and guardrails
├── test_cli.py           # Command-line test (no WhatsApp needed)
├── rag_data/             # Knowledge base documents
│   ├── food-guide.docx                    # Massachusetts WIC Approved Food Guide
│   ├── Cold Food Storage Chart.pdf        # USDA cold food storage times
│   ├── food_safety_knowledge_base.txt     # Food safety reference text
│   └── wic_stores_near_tufts_medford.txt  # WIC store list near Tufts/Medford
└── user_memory/          # Auto-created; stores per-user memory files
```

---

## RAG Pipeline

The RAG system (`rag_pipeline.py`) gives the bot the ability to answer questions grounded in real documents rather than relying solely on the LLM's training data.

### How it works

```
Documents (PDF / DOCX / TXT)
        ↓  load + extract text
        ↓  split into ~450-char chunks (50-char overlap)
        ↓  embed with sentence-transformers (all-MiniLM-L6-v2)
        ↓  store in FAISS vector index
                        ↓
User question  →  embed question  →  top-3 nearest chunks
                        ↓
        Inject chunks as CONTEXT into prompt
                        ↓
        LLM generates a grounded answer
```

### Knowledge base documents

| File | Content |
|---|---|
| `food-guide.docx` | Massachusetts WIC Approved Food Guide — lists allowed foods by category (milk, grains, produce, etc.) |
| `Cold Food Storage Chart.pdf` | USDA-recommended safe refrigerator and freezer storage times for common foods |
| `food_safety_knowledge_base.txt` | Food handling best practices, safe cooking temperatures, and illness prevention |
| `wic_stores_near_tufts_medford.txt` | WIC-authorized grocery stores in the Medford/Somerville area |

### Enabling RAG

In `Nutrition_Bot.py`, set the toggle to `True`:

```python
USE_RAG = True   # default is False
```

When `True`, every WhatsApp message goes through the full RAG pipeline before reaching the LLM.

### Scope guardrail (pre-filter)

Before calling the main LLM, `rag_pipeline.py` runs a lightweight scope check:

```python
def is_in_scope(self, question: str) -> bool:
    # Asks a classifier LLM: "Is this about human nutrition/food safety? YES or NO"
    # Returns False → bot immediately declines without running the full pipeline
```

This prevents the bot from answering out-of-scope questions (e.g. pet nutrition, non-food topics) even if the LLM would otherwise try to be helpful.

---

## Per-User Memory

The `UserMemory` class (`user_memory.py`) automatically extracts and stores structured user profile information from each conversation turn.

### What gets stored

After every message, the bot runs a structured extraction to detect and save:

| Field | Example |
|---|---|
| `name` | Sarah |
| `age_group` | child / adult / elder |
| `gender` | female |
| `asking_for` | self / child / parent / spouse |
| `health_conditions` | diabetes, pregnancy, heart disease |
| `allergies` | peanuts, dairy, shellfish |
| `medications` | metformin, warfarin |
| `dietary_restriction` | vegetarian, gluten-free, halal |
| `disliked_foods` | broccoli, spicy food |
| `main_goal` | lose weight, manage blood sugar |

### Storage format

Each user gets a plain-text file at `user_memory/<user_id>.txt`. New facts are appended on each turn. A lightweight FAISS index is built on demand so relevant memories can be retrieved for any question.

### How memory improves answers

At query time, `get_context()` combines both sources:

```
[Public Knowledge Base]   ← from rag_data/ documents
[User Memory]             ← from user_memory/<user_id>.txt
         ↓
   Injected into prompt → LLM gives a personalized answer
```

---

## Google Places Integration — WIC Store Finder

The `LocationService` class (`location_service.py`) finds WIC-authorized grocery stores near the user using the Google Places API.

### How it works

1. **Input**: user's GPS coordinates (from a WhatsApp location message) or a text address
2. **Geocoding** (address only): converts text address → `(lat, lng)` via Google Geocoding API
3. **Nearby Search**: queries Google Places API for grocery stores within a configurable radius (default: 3 miles)
4. **WIC cross-reference**: filters and prioritizes results against a known list of WIC-authorized store chains (Stop & Shop, Market Basket, Star Market, Shaw's, Hannaford, CVS, Walgreens, etc.)
5. **Output**: formatted reply with store name, address, distance, open/closed status, rating, and a Google Maps link

### API endpoints used

| API | Purpose |
|---|---|
| `maps.googleapis.com/maps/api/place/nearbysearch` | Find grocery stores near coordinates |
| `maps.googleapis.com/maps/api/geocode` | Convert text address to coordinates |

### Setup

Add your Google Places API key to `.env`:

```env
GOOGLE_PLACES_API_KEY=your_key_here
```

To get a key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Places API** and **Geocoding API**
3. Create an API key under **APIs & Services → Credentials**

### Example output

```
Here are WIC-authorized grocery stores near you:

1. Stop & Shop ✓ Open now
   123 Main St, Somerville
   0.8 miles away  Rating: 4.1/5
   Map: https://www.google.com/maps/place/?q=place_id:...

2. Market Basket ✓ Open now
   456 Broadway, Medford
   1.2 miles away  Rating: 4.4/5
   Map: https://www.google.com/maps/place/?q=place_id:...

Tip: Always bring your WIC card and check with the store about
which specific items are covered before shopping.
```

---

## Environment Variables

Create a `.env` file in the `NutritionBot/` directory:

```env
# ngrok (required to expose local server to WhatsApp)
NGROK_AUTH_TOKEN=your_ngrok_token
PORT=8000

# LLMProxy (required for all LLM calls)
LLMPROXY_ENDPOINT=https://your-llmproxy-endpoint
LLMPROXY_API_KEY=your_llmproxy_api_key

# Google Places (required for WIC store finder)
GOOGLE_PLACES_API_KEY=your_google_places_api_key
```

---

## Running the Bot

```bash
cd NutritionBot

# Install dependencies
pip install -r requirements_rag.txt

# Start the WhatsApp webhook server
python Main.py
```

Once running, the terminal will print a public Webhook URL. Register this URL in your WhatsApp / messaging platform dashboard.

## Testing Locally (no WhatsApp needed)

```bash
# Simple CLI chat (RAG disabled)
python test_cli.py

# Full RAG interactive test
python rag_pipeline.py

# WIC store finder test
python location_service.py
```
<<<<<<< HEAD

## Onboarding Script

You can run a simple interactive onboarding flow that stores structured profile data in `user_memory/<user_id>.txt`:

```bash
python onboarding.py
```

Fields captured:
   - name, age_group, gender, health_conditions, allergies, medications,
      asking_for, main_goal
=======
>>>>>>> origin/main
