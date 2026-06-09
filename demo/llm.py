"""Bedrock Claude wrapper (Converse API, streaming) for the chat demo."""
import boto3

MODEL_ID = "us.anthropic.claude-opus-4-1-20250805-v1:0"
REGION = "us-west-2"

_client = None


def client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    return _client


def stream_chat(messages, system, max_tokens=1500, temperature=0.4):
    """Yield text deltas from Claude. `messages`: [{role, content(str)}]."""
    conv = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
    resp = client().converse_stream(
        modelId=MODEL_ID,
        messages=conv,
        system=[{"text": system}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    for event in resp["stream"]:
        if "contentBlockDelta" in event:
            d = event["contentBlockDelta"]["delta"]
            if "text" in d:
                yield d["text"]


def complete(prompt, system, max_tokens=1200, temperature=0.4):
    r = client().converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": system}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return r["output"]["message"]["content"][0]["text"]
