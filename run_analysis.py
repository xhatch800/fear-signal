 import anthropic
import json
import os
import time
from datetime import datetime

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")

print(f"Running analysis for {today}...")

def api_call(messages, tools=None, max_retries=4):
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model="claude-haiku-4-5",
                max_tokens=3000,
                messages=messages,
            )
            if tools:
                kwargs["tools"] = tools
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Exceeded retries")

# Step 1: search and summarise in plain text
SEARCH_PROMPT = f"""Today is {today}. Use web search to find today's top news headlines across major outlets.

Then write a plain-text summary of the 5 most widely circulating CONTESTED fears in the media today — fears where reasonable people genuinely disagree about the threat level. For each one describe: what the fear is, why it is contested, the strongest case for it being real, the strongest case for it being exaggerated, and 3 reflection questions. Also briefly name 5 more fears worth mentioning.

Write clearly and completely. Do not use JSON yet."""

messages = [{"role": "user", "content": SEARCH_PROMPT}]

# Run search loop
for attempt in range(8):
    response = api_call(messages, tools=[{"type": "web_search_20250305", "name": "web_search"}])
    messages.append({"role": "assistant", "content": response.content})

    print(f"Step1 attempt {attempt}: stop={response.stop_reason}, blocks={[b.type for b in response.content]}")

    if response.stop_reason == "end_turn":
        text_block = next((b for b in response.content if b.type == "text" and b.text.strip()), None)
        if text_block:
            summary_text = text_block.text
            print(f"Got summary ({len(summary_text)} chars)")
            break
        else:
            raise ValueError("end_turn with no text")
    elif response.stop_reason == "tool_use":
        continue
    else:
        raise ValueError(f"Unexpected stop_reason: {response.stop_reason}")
else:
    raise ValueError("Too many search iterations")

# Step 2: convert summary to JSON in a fresh call (no tools)
FORMAT_PROMPT = f"""Convert the following analysis into valid JSON. Output ONLY the JSON object, nothing else — no markdown, no backticks, no explanation.

Analysis to convert:
{summary_text}

Required JSON format:
{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "Neutral 4-6 word title",
      "reach": "One sentence on circulation.",
      "the_fear": "One neutral sentence.",
      "why_contested": "Two sentences.",
      "case_for": "One sentence.",
      "case_against": "One sentence.",
      "questions": ["Q1?", "Q2?", "Q3?"]
    }}
  ],
  "honorable_mentions": [
    {{ "title": "Short title", "summary": "One sentence." }}
  ]
}}"""

format_response = api_call([{"role": "user", "content": FORMAT_PROMPT}])
raw = next((b.text for b in format_response.content if b.type == "text"), "")

print(f"Raw JSON output (first 300): {raw[:300]}")

raw = raw.strip().replace("```json", "").replace("```", "").strip()
start = raw.find("{")
end = raw.rfind("}") + 1
raw = raw[start:end]

parsed = json.loads(raw)

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Done. Saved {len(parsed.get('top5', []))} fears.")
