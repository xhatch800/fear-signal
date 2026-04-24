import anthropic
import json
import os
import sys
import time
from datetime import datetime

def log(msg):
    print(msg, flush=True)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")

log(f"START: Running analysis for {today}")

PROMPT = f"""Today is {today}. Search today's news headlines and identify 5 widely circulating contested fears where reasonable people disagree about the threat level. Also list 5 brief honorable mentions.

Respond ONLY with a JSON object. Start with {{ and end with }}. No markdown, no backticks, no explanation:

{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "4-6 word neutral title",
      "reach": "One sentence on circulation.",
      "the_fear": "One neutral sentence.",
      "why_contested": "Two sentences on uncertainty.",
      "case_for": "One sentence why some find it legitimate.",
      "case_against": "One sentence why others find it exaggerated.",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
  ],
  "honorable_mentions": [
    {{"title": "Short title", "summary": "One sentence."}}
  ]
}}"""

messages = [{"role": "user", "content": PROMPT}]
all_text = []

for turn in range(10):
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2500,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=messages,
            )
            break
        except anthropic.RateLimitError:
            wait = 40 * (attempt + 1)
            log(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError("Rate limit retries exhausted")

    block_types = [b.type for b in resp.content]
    log(f"Turn {turn}: stop={resp.stop_reason} blocks={block_types}")

    for block in resp.content:
        t = getattr(block, 'text', None)
        if t and t.strip():
            log(f"  TEXT: {repr(t[:200])}")
            all_text.append(t)

    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        log("  end_turn reached")
        break

combined = " ".join(all_text).strip()
log(f"Total text: {len(combined)} chars")

if not combined:
    raise ValueError("No text returned from API")

start = combined.find("{")
end = combined.rfind("}") + 1
if start == -1 or end <= 1:
    raise ValueError(f"No JSON found. Raw: {repr(combined[:400])}")

raw = combined[start:end]
log(f"Parsing JSON ({len(raw)} chars)...")

parsed = json.loads(raw)
os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

log(f"SUCCESS: Saved {len(parsed.get('top5', []))} fears to data/today.json")
