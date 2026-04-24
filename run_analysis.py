import anthropic
import json
import os
import time
from datetime import datetime

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")

print(f"Running analysis for {today}...")

def call(messages, use_search=False, max_retries=5):
    for attempt in range(max_retries):
        try:
            kwargs = dict(model="claude-haiku-4-5", max_tokens=2000, messages=messages)
            if use_search:
                kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
            resp = client.messages.create(**kwargs)
            return resp
        except anthropic.RateLimitError as e:
            wait = 45 * (attempt + 1)
            print(f"  Rate limited (attempt {attempt+1}). Waiting {wait}s... error: {e}")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(15)
    raise RuntimeError("All retries exhausted")

# Single prompt that searches AND returns JSON directly
PROMPT = f"""Today is {today}.

Search today's news. Find 5 widely circulating contested fears where reasonable people disagree. Also find 5 more brief ones.

You MUST respond with ONLY a JSON object. No text before or after. No markdown. No backticks. Begin your response with {{ and end with }}.

{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "4-6 word neutral title",
      "reach": "One sentence on how widely this is circulating.",
      "the_fear": "One neutral sentence stating the fear.",
      "why_contested": "Two sentences on genuine uncertainty.",
      "case_for": "One sentence why some find it legitimate.",
      "case_against": "One sentence why others find it exaggerated.",
      "questions": ["Reflection question 1?", "Reflection question 2?", "Reflection question 3?"]
    }}
  ],
  "honorable_mentions": [
    {{"title": "Short title", "summary": "One sentence."}}
  ]
}}"""

messages = [{"role": "user", "content": PROMPT}]

text = ""
for turn in range(10):
    resp = call(messages, use_search=True)
    print(f"Turn {turn}: stop={resp.stop_reason} types={[b.type for b in resp.content]}")
    
    messages.append({"role": "assistant", "content": resp.content})
    
    # Collect any text blocks
    for block in resp.content:
        if hasattr(block, 'text') and block.text.strip():
            text += block.text
            print(f"  Got text block: {repr(block.text[:150])}")
    
    if resp.stop_reason == "end_turn":
        break

print(f"Total text collected: {len(text)} chars")
print(f"First 300 chars: {repr(text[:300])}")

if not text.strip():
    raise ValueError("No text returned from API after all turns")

# Extract JSON
text = text.strip()
start = text.find("{")
end = text.rfind("}") + 1
if start == -1 or end <= 1:
    raise ValueError(f"No JSON object found. Got: {repr(text[:300])}")

raw = text[start:end]
print(f"Extracted JSON ({len(raw)} chars), parsing...")

parsed = json.loads(raw)

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"SUCCESS. Saved {len(parsed.get('top5', []))} fears to data/today.json")
