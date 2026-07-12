"""LLM-powered entity and relationship extraction using Google Gemini."""

import json
import re
import time
from google import genai


EXTRACTION_PROMPT = """You are an expert at extracting structured knowledge from text.
Given the following text, extract ALL entities and relationships.

TEXT:
{text}

Return a valid JSON object with this EXACT structure:
{{
  "entities": [
    {{"name": "ENTITY_NAME", "type": "person|organization|technology|algorithm|concept|tool|dataset|event", "description": "one-line description"}}
  ],
  "relationships": [
    {{"source": "ENTITY_A", "target": "ENTITY_B", "description": "how A relates to B", "weight": 1}}
  ]
}}

Rules:
- Entity names should be UPPERCASE
- Extract ALL people, organizations, technologies, algorithms mentioned
- Include relationships like "created", "works_at", "is_part_of", "used_for", etc.
- Return ONLY valid JSON, no markdown fences
"""


def call_gemini(prompt: str, api_key: str, retries: int = 3) -> str | None:
    """Call Gemini API with exponential backoff retry logic.

    Args:
        prompt: The prompt to send.
        api_key: Google Gemini API key.
        retries: Number of retry attempts on failure.

    Returns:
        Raw response text, or None if all retries fail.
    """
    client = genai.Client(api_key=api_key)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            return response.text
        except Exception as e:
            print(f"  [Retry {attempt+1}/{retries}] API error: {str(e)[:80]}")
            time.sleep(5 * (attempt + 1))

    return None


def extract_entities_relationships(text: str, api_key: str) -> dict:
    """Extract entities and relationships from text using Gemini.

    Args:
        text: Input text to analyze.
        api_key: Google Gemini API key.

    Returns:
        Dict with 'entities' and 'relationships' lists.
    """
    raw = call_gemini(EXTRACTION_PROMPT.format(text=text), api_key)
    if not raw:
        return {"entities": [], "relationships": []}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"entities": [], "relationships": []}
