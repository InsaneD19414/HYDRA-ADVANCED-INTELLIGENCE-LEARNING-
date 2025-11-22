# agent_template.py
# Simple agent template using FastAPI + Redis-backed task queue (RQ).
# Each agent instance performs one repeatable income task (e.g., content generation, affiliate outreach).
import os
import time
import json
import random
from fastapi import FastAPI, BackgroundTasks
from redis import Redis
from rq import Queue
from pydantic import BaseModel
import requests

app = FastAPI()
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_conn = Redis.from_url(redis_url)
q = Queue(connection=redis_conn)

# Basic config driven by env
AGENT_ROLE = os.getenv("AGENT_ROLE", "content_writer")
AGENT_ID = os.getenv("AGENT_ID", "agent-{{RANDOM_ID}}")
PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH", "/app/prompts/prompt.md")

class TaskRequest(BaseModel):
    input_data: dict = {}

def perform_task(agent_id, role, input_data):
    """
    Replace this function with calls to your chosen LLM API (OpenAI, local LLM, etc).
    Keep API keys in secrets manager. This function must be idempotent and log to central store.
    """
    # load prompt template
    try:
        with open(PROMPT_TEMPLATE_PATH, "r") as f:
            prompt = f.read()
    except:
        prompt = f"Role: {role}. Input: {input_data}"
    # Simulate work
    result = {
        "agent_id": agent_id,
        "role": role,
        "prompt_excerpt": prompt[:200],
        "payload": input_data,
        "value_estimate": random.random()
    }
    # publish to central results endpoint (replace with your webhook)
    try:
        requests.post(os.getenv("RESULTS_WEBHOOK", "http://monitor:8001/results"), json=result, timeout=5)
    except Exception as e:
        print("Monitor publish failed:", e)
    return result

@app.post("/enqueue")
def enqueue_task(req: TaskRequest, background_tasks: BackgroundTasks):
    job = q.enqueue(perform_task, AGENT_ID, AGENT_ROLE, req.input_data)
    return {"enqueued": True, "job_id": job.get_id()}

@app.get("/health")
def health():
    return {"status": "ok", "agent_id": AGENT_ID, "role": AGENT_ROLE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
