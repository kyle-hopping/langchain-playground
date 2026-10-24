from config import model

# streaming allows you to receive incremental data, token deltas, and step 
# updates from agents and language models in real time rather than waiting 
# for a full response.
for chunk in model.stream("Write a structured essay about OpenClaw. Keep it to 2 paragraphs."):
    # print(chunk.content, end="", flush=True)
    print(repr(chunk.content))
print()