import anthropic
import json
import os
import time
from datetime import datetime

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")

print(f"Running analysis for {today}...")

def api_call_with_search(prompt, max_retries=4):
    """Single call with web search — handles all turn-taking internally."""
    for attempt in range(max_retries):
        try:
            # Use streaming to get full response including after tool use
            messages = [{"role": "user", "content": prompt}]
            
            # Keep looping until end_turn
            for _ in range(10):
                resp = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=3000,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    messages=messages,
                )
                
                # Log what we got
                print(f"  Response: stop={resp.stop_reason} blocks={[b.type for b in resp.content]}")
                
                # Append assistant turn
                messages.append({"role": "assistant", "content": resp.content})
                
                if resp.stop_reason == "end_turn":
                    # Find any text block
                    for block in resp.content:
                        if hasattr(block, 'text') and block.text.strip():
                            return block.text
                    # No text — shouldn't happen but handle it
                    print("  end_turn but no text block, continuing...")
                    return None
                
                # For tool_use or any other reason, just continue the loop
                # The web search tool handles its own results server-side
                # We just need to pass back the assistant message and continue
                # No user tool_result needed for web_search_20250305
                
            return None
            
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Exceeded retries")

def api_call_plain(prompt, max_retries=4):
    """Simple call with no tools."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if hasattr(block, 'text') and block.text.strip():
                    return block.text
            return ""
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Exceeded retries")

# Step 1: search and summarise
SEARCH_PROMPT = f"""Today is {today}. Search the web for today's major news headlines.

Then identify and describe the 5 most widely circulating CONTESTED fears in media today — fears where reasonable people genuinely disagree. For each: state the fear, explain why it is contested, give the strongest case it is real, give the strongest case it is exaggerated, and list 3 reflection questions. Also briefly name 5 more fears.

Write in plain English. Be thorough."""

summary = api_call_with_search(SEARCH_PROMPT)
if not summary:
    raise ValueError("No summary returned from search step")

print(f"Got summary ({len(summary)} chars). First 200: {summary[:200]}")

# Step 2: format as JSON
FORMAT_PROMPT = f"""Convert this analysis to JSON. Output ONLY the JSON — no markdown, no backticks, no explanation.

Analysis:
{summary}

JSON format:
{{
  "date": "{today}",
  "top5": [
    {{
      "rank": 1,
      "title": "Neutral 4-6 word title",
      "reach": "One sentence.",
      "the_fear": "One sentence.",
      "why_contested": "Two sentences.",
      "case_for": "One sentence.",
      "case_against": "One sentence.",
      "questions": ["Q1?", "Q2?", "Q3?"]
    }}
  ],
  "honorable_mentions": [{{ "title": "Title", "summary": "One sentence." }}]
}}"""

raw = api_call_plain(FORMAT_PROMPT)
print(f"Raw JSON (first 200): {raw[:200]}")

raw = raw.strip().replace("```json","").replace("```","").strip()
start = raw.find("{")
end = raw.rfind("}") + 1
if start == -1 or end == 0:
    raise ValueError(f"No JSON found in: {raw[:300]}")
raw = raw[start:end]

parsed = json.loads(raw)
os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(parsed, f, indent=2)

print(f"Done. Saved {len(parsed.get('top5', []))} fears.")
