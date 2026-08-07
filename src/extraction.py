"""LLM-powered entity and relationship extraction using Mistral AI."""

import json
import re
import time
from mistralai.client import Mistral


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


def call_mistral(prompt: str, api_key: str, retries: int = 3) -> str | None:
    """Call Mistral API with exponential backoff retry logic.

    Args:
        prompt: The prompt to send.
        api_key: Mistral API key.
        retries: Number of retry attempts on failure.

    Returns:
        Raw response text, or None if all retries fail.
    """
    client = Mistral(api_key=api_key)

    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [Retry {attempt+1}/{retries}] API error: {str(e)[:80]}")
            time.sleep(5 * (attempt + 1))

    return None


def extract_entities_relationships(text: str, api_key: str) -> dict:
    """Extract entities and relationships from text using Mistral.

    Args:
        text: Input text to analyze.
        api_key: Mistral API key.

    Returns:
        Dict with 'entities' and 'relationships' lists.
    """
    raw = call_mistral(EXTRACTION_PROMPT.format(text=text), api_key)
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
