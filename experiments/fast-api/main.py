# main.py
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ── Simulated "database" ──────────────────────────────────────────
DATA_STORE = {
    "alpha": "first letter",
    "beta":  "second letter",
    "gamma": "third letter",
}

# ── Backend functions (pure Python, no FastAPI coupling) ──────────
def post_fn(user_input: str) -> str:
    return user_input.upper()

def get_fn(user_input: str) -> str:
    value = DATA_STORE.get(user_input)
    if value is None:
        raise KeyError(f"Key '{user_input}' not found")
    return value


# ── Request/Response models ───────────────────────────────────────
class PostRequest(BaseModel):
    user_input: str          # Pydantic validates + coerces this from JSON body

class PostResponse(BaseModel):
    status: str
    result: str

class GetResponse(BaseModel):
    status: str
    key: str
    value: str


# ── Routes ────────────────────────────────────────────────────────
@app.post("/transform", response_model=PostResponse)
def handle_post(body: PostRequest):
    """
    Body parameter: FastAPI sees a Pydantic model → expects JSON body.
    If body parsing fails, FastAPI auto-returns 422 Unprocessable Entity.
    """
    result = post_fn(body.user_input)
    return PostResponse(status="success", result=result)


@app.get("/lookup", response_model=GetResponse)
def handle_get(user_input: str = Query(..., description="Key to look up")):
    """
    Query parameter: primitive type annotation → FastAPI reads from ?user_input=
    Query(...) means required; Query("default") would make it optional.
    """
    try:
        value = get_fn(user_input)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return GetResponse(status="success", key=user_input, value=value)