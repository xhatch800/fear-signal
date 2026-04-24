import anthropic
import json
import os
import time
from datetime import datetime

client = anthropic.Anthropic()

today = datetime.now().strftime("%B %d, %Y")

PROMPT = f"""Today is {today}. Search the news and identify the 5 most widely circulating contested fears in media right now — where reasonable people genuinely disagree about the threat. Also list 5 honorable mentions.

Respond ONLY with valid JSON, no markdown:
{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "Neutral 4-6 word title",
      "reach": "One sentence on how widely circulating.",
      "the_fear": "One neutral sentence stating the fear.",
      "why_contested": "Two sentences on genuine uncertainty.",
      "case_for": "One sentence: why some find it legitimate.",
      "case_against": "One sentence: why others find it exaggerated.",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
  ],
  "honorable_mentions": [
    {{ "title": "Short title", "summary": "One sentence." }}
  ]
}}"""

print(f"Running analysis for {today}...")

def run_with_retry(messages, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=3000,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages
            )
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}...")
            time.sleep(wait)
    raise RuntimeError("Exceeded retries due to rate limiting")

messages = [{"role": "user", "content": PROMPT}]

for attempt in range(6):
    response = run_with_retry(messages)
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        text_block = next((b for b in response.content if b.type == "text"), None)
        if text_block and text_block.text.strip():
            break
        else:
            raise ValueError("end_turn reached but no text block found")
    elif response.stop_reason == "tool_use":
        continue
    else:
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
else:
    raise ValueError("Exceeded maximum loop attempts")

raw = text_block.text.strip().replace("```json", "").replace("```", "").strip()
parsed = json.loads(raw)

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Saved {len(parsed.get('top5', []))} fears to data/today.json")
