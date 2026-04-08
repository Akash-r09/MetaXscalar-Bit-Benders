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
        "reward": 0.01,
        "done": False
    }

@app.post("/step")
async def step(action: Action):
    state["step"] += 1

    msg = action.message
    task = state["task"]

    # task-specific scaling
    if task == "echo_easy":
        base = len(msg) / 1000
    elif task == "echo_medium":
        base = len(msg) / 1400
    else:  # echo_hard
        base = len(msg) / 1800

    reward = 0.2 + base
    reward = min(reward, 0.9)

    reward = max(0.05, min(reward, 0.95))   

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