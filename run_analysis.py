import anthropic
import json
import os
import time
from datetime import datetime

def log(msg):
    print(msg, flush=True)

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")
log("START: " + today)

prompt = ("Today is " + today + ". Search today's news. Find 5 widely circulating CONTESTED fears "
          "where reasonable people genuinely disagree about the threat level. Also list 5 honorable mentions.\n\n"
          "After searching, respond ONLY with a JSON object. No markdown, no backticks, no explanation.\n\n"
          '{"date":"' + today + '",'
          '"top5":[{"rank":1,"title":"4-6 word neutral title","reach":"One sentence.","the_fear":"One sentence.",'
          '"why_contested":"Two sentences.","case_for":"One sentence.","case_against":"One sentence.",'
          '"questions":["Q1?","Q2?","Q3?"]}],'
          '"honorable_mentions":[{"title":"Title","summary":"One sentence."}]}')

messages = [{"role": "user", "content": prompt}]
final_json = None

for turn in range(15):
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
            log("Rate limited, waiting 40s...")
            time.sleep(40)
    else:
        raise RuntimeError("Retries exhausted")

    log("Turn " + str(turn) + " stop=" + str(resp.stop_reason) + " blocks=" + str(len(resp.content)))
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        # Find text block
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                text = block.text.strip()
                log("Text: " + repr(text[:120]))

                # Strip markdown fences
                if "```" in text:
                    for part in text.split("```")[1:]:
                        part = part.strip()
                        if part.startswith("json"):
                            part = part[4:].strip()
                        if part.startswith("{"):
                            text = part
                            break

                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > 1:
                    try:
                        final_json = json.loads(text[start:end])
                        log("JSON parsed successfully!")
                        break
                    except json.JSONDecodeError as e:
                        log("JSON parse failed: " + str(e))

                # Not JSON yet — nudge the model to produce it
                log("No JSON found, nudging model...")
                messages.append({
                    "role": "user",
                    "content": "Good. Now respond with ONLY the JSON object based on what you found. No markdown, no backticks, start with { and end with }."
                })
                break

        if final_json:
            break

if not final_json:
    raise ValueError("Failed to get valid JSON after all turns")

os.makedirs("data", exist_ok=True)
with open("data/today.json", "w") as f:
    json.dump(final_json, f, indent=2)

log("SUCCESS: Saved " + str(len(final_json.get("top5", []))) + " fears")
