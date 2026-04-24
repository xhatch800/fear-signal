import anthropic
import json
import os
from datetime import datetime

client = anthropic.Anthropic()

today = datetime.now().strftime("%B %d, %Y")

PROMPT = f"""You are a calm, thoughtful guide helping people navigate today's media with more clarity. Your job is not to tell people what to think — it is to help them see where genuine uncertainty exists so they can pause before taking sides.

Using web search, scan today's top news stories and identify the 5 most prominent fear-based narratives being circulated, plus 5 additional ones worth noting. Focus specifically on fears that are contested — meaning reasonable, informed people genuinely disagree about whether the threat is real, exaggerated, or misrepresented.

Today's date is {today}.

Rank them in order of how widely they are circulating today — most pervasive first. Base this on how many major outlets are covering the story, how recently coverage spiked, and how broadly it crosses different types of media sources.

For each of the top 5 contested fears, use this structure:

Title: Give the fear a short, neutral, plain-language title — 4 to 6 words maximum. No sensationalism, no bias toward either side.

Reach: In one sentence, describe how widely this fear is currently circulating — which types of outlets are covering it, how recently it spiked, and whether it is growing or settling.

The Fear: State the fear plainly and neutrally, in one sentence, exactly as it is circulating in media.

Why It's Contested: In 2 to 3 sentences, explain what is genuinely uncertain. What do we actually know? What is still unresolved? Avoid taking sides.

The Strongest Cases: Present the most honest version of why some people find this fear legitimate, and why others find it exaggerated or unfounded. Give each side equal weight and equal respect.

Questions Worth Sitting With: Offer 2 to 3 genuine reflection questions — not rhetorical, not leading — that help the reader stay curious rather than reactive.

For the 5 honorable mentions, provide just a short title and one-sentence description.

Respond ONLY with valid JSON, no markdown, no preamble, no explanation:
{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "Short neutral title 4-6 words",
      "reach": "One sentence on circulation.",
      "the_fear": "The fear in one neutral sentence.",
      "why_contested": "2-3 sentences on genuine uncertainty.",
      "case_for": "Honest case for why people find it legitimate.",
      "case_against": "Honest case for why others find it exaggerated.",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
  ],
  "honorable_mentions": [
    {{ "title": "Short title", "summary": "One sentence." }}
  ]
}}"""

print(f"Running analysis for {today}...")

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=3000,
    tools=[{"type": "web_search_20250305", "name": "web_search"}],
    messages=[{"role": "user", "content": PROMPT}]
)

text_block = next((b for b in response.content if b.type == "text"), None)
if not text_block:
    raise ValueError("No text block returned from API")

raw = text_block.text.strip().replace("```json", "").replace("```", "").strip()
parsed = json.loads(raw)

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Saved {len(parsed.get('top5', []))} fears to data/today.json")
