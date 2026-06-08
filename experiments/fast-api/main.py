# main.py
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import mlflow
from typing import Optional
from google.protobuf.json_format import MessageToDict

mlflow.set_tracking_uri("databricks://joshuale-common")  # hard-code to test
app = FastAPI()

# ── Simulated "database" ──────────────────────────────────────────
DATA_STORE = {
    "alpha": "first letter",
    "beta": "second letter",
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
    user_input: str  # Pydantic validates + coerces this from JSON body


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


@app.get("/search_experiments")
def search_experiments(
    view_type: str = "ACTIVE_ONLY",
    max_results: Optional[int] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[list[str]] = None,
):
    try:
        # vt = getattr(mlflow.entities.ViewType, view_type, None)
        allowed_vt = [k.upper() for k in mlflow.entities.ViewType._STRING_TO_VIEW.keys()]
        if view_type not in allowed_vt:
            raise HTTPException(400, detail=f"Invalid view_type: '{view_type}'. Allowed view_types: {allowed_vt}") 
        
        experiments = mlflow.search_experiments(
            view_type=getattr(mlflow.entities.ViewType, view_type),
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by,
        )

        dicts = [
            MessageToDict(e.to_proto(), preserving_proto_field_name=True)
            for e in experiments
        ]
    
    except mlflow.exceptions.MlflowException as e:
        raise HTTPException(502, detail=str(e))
    
    return dicts