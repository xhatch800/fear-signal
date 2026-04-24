import anthropic, json, os, sys, time
from datetime import datetime

def log(msg):
    print(msg, flush=True)

client = anthropic.Anthropic()
today = datetime.now().strftime("%B %d, %Y")
log("START: " + today)

prompt = "Today is " + today + ". Search news. Find 5 contested media fears and 5 honorable mentions. Reply ONLY with JSON, no markdown: {\"date\":\"today\",\"top5\":[{\"rank\":1,\"title\":\"x\",\"reach\":\"x\",\"the_fear\":\"x\",\"why_contested\":\"x\",\"case_for\":\"x\",\"case_against\":\"x\",\"questions\":[\"q1\",\"q2\",\"q3\"]}],\"honorable_mentions\":[{\"title\":\"x\",\"summary\":\"x\"}]}"

messages = [{"role": "user", "content": prompt}]

for turn in range(10):
    for attempt in range(4):
        try:
            resp = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=2500, tools=[{"type": "web_search_20250305", "name": "web_search"}], messages=messages)
            break
        except anthropic.RateLimitError:
            log("Rate limited, waiting 40s...")
            time.sleep(40)
    else:
        raise RuntimeError("Retries exhausted")

    log("Turn " + str(turn) + " stop=" + str(resp.stop_reason) + " blocks=" + str(len(resp.content)))
    for i, b in enumerate(resp.content):
        log("  [" + str(i) + "] type=" + str(b.type))
        for attr in ["text","content","input","name","id"]:
            v = getattr(b, attr, None)
            if v: log("    ." + attr + "=" + repr(str(v)[:150]))

    messages.append({"role": "assistant", "content": resp.content})
    if resp.stop_reason == "end_turn":
        log("DONE")
        break

log("Script finished - check Turn lines above")
