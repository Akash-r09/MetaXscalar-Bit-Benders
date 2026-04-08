from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import random

app = FastAPI()

class Action(BaseModel):
    message: str

TASKS = ["echo_easy", "echo_medium", "echo_hard"]

state = {
    "step": 0,
    "done": False,
    "task": "echo_easy"
}

@app.post("/reset")
async def reset():
    state["step"] = 0
    state["done"] = False
    state["task"] = random.choice(TASKS)

    return {
        "observation": {
            "echoed_message": f"start ({state['task']})"
        },
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step(action: Action):
    state["step"] += 1

    msg = action.message
    task = state["task"]

    if task == "echo_easy":
        reward = min(0.2 + len(msg)/1200, 0.9)

    elif task == "echo_medium":
        reward = min(0.2 + len(msg)/1200, 0.9)

    else:  # echo_hard
        reward = min(0.2 + len(msg)/1200, 0.9)

    done = state["step"] >= 5

    return {
        "observation": {"echoed_message": msg},
        "reward": reward,
        "done": done
    }

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()