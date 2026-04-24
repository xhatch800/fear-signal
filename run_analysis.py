import anthropic
import json
import os
from datetime import datetime

client = anthropic.Anthropic()

today = datetime.now().strftime("%B %d, %Y")

PROMPT = f"""You are a calm guide helping people pause before reacting to today's media. Today is {today}.

Use web search to find today's top news. Identify the 5 most widely circulating CONTESTED fears — where reasonable people genuinely disagree about the threat level. Rank by pervasiveness. Also list 5 brief honorable mentions.

For each of the top 5, be concise — 1-2 sentences per field.

Respond ONLY with valid JSON, no markdown, no preamble:
{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "Neutral title 4-6 words",
      "reach": "One sentence: which outlets, how widespread.",
      "the_fear": "One neutral sentence stating the fear.",
      "why_contested": "2 sentences on genuine uncertainty.",
      "case_for": "1-2 sentences: why some find it legitimate.",
      "case_against": "1-2 sentences: why others find it exaggerated.",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
  ],
  "honorable_mentions": [
    {{ "title": "Short title", "summary": "One sentence." }}
  ]
}}"""

print(f"Running analysis for {today}...")

messages = [{"role": "user", "content": PROMPT}]

for attempt in range(5):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=messages
    )

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
    raise ValueError("Exceeded maximum attempts waiting for final response")

raw = text_block.text.strip().replace("```json", "").replace("```", "").strip()
parsed = json.loads(raw)

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Saved {len(parsed.get('top5', []))} fears to data/today.json")
