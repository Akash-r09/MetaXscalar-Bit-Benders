from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Action(BaseModel):
    message: str

state = {
    "step": 0,
    "done": False
}

@app.post("/reset")
async def reset():
    state["step"] = 0
    state["done"] = False
    return {
        "observation": {"echoed_message": "start"},
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
async def step(action: Action):
    state["step"] += 1

    msg = action.message
    reward = len(msg) * 0.1

    done = state["step"] >= 8

    return {
        "observation": {"echoed_message": msg},
        "reward": reward,
        "done": done
    }