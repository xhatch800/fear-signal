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
          "Respond ONLY with a JSON object. No markdown, no backticks, no explanation.\n\n"
          '{"date":"' + today + '",'
          '"top5":[{"rank":1,"title":"4-6 word neutral title","reach":"One sentence.","the_fear":"One sentence.",'
          '"why_contested":"Two sentences.","case_for":"One sentence.","case_against":"One sentence.",'
          '"questions":["Q1?","Q2?","Q3?"]}],'
          '"honorable_mentions":[{"title":"Title","summary":"One sentence."}]}')

messages = [{"role": "user", "content": prompt}]

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
            log("Rate limited, waiting 40s...")
            time.sleep(40)
    else:
        raise RuntimeError("Retries exhausted")

    log("Turn " + str(turn) + " stop=" + str(resp.stop_reason) + " blocks=" + str(len(resp.content)))
    messages.append({"role": "assistant", "content": resp.content})

    if resp.stop_reason == "end_turn":
        # Find the text block
        text = ""
        for block in resp.content:
            if block.type == "text" and block.text.strip():
                text = block.text
                log("Got text block: " + repr(text[:100]))
                break

        if not text:
            raise ValueError("end_turn but no text block found")

        # Strip markdown fences robustly
        text = text.strip()
        if "```" in text:
            # Extract content between first ``` and last ```
            parts = text.split("```")
            # parts[1] will be "json\n{...}" or just "{...}"
            for part in parts[1:]:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break

        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= 1:
            raise ValueError("No JSON found in: " + repr(text[:200]))

        raw = text[start:end]
        log("Parsing JSON (" + str(len(raw)) + " chars)...")

        parsed = json.loads(raw)
        os.makedirs("data", exist_ok=True)
        with open("data/today.json", "w") as f:
            json.dump(parsed, f, indent=2)

        log("SUCCESS: Saved " + str(len(parsed.get("top5", []))) + " fears to data/today.json")
        break
