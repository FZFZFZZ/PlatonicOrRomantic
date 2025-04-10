from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from models.predictor import LstmPredictor
from models.explainer import LimeExplainer
from .types import RequestModel, ResponseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()

predictor = LstmPredictor(variant="glove.6B.50d")
explainer = LimeExplainer(predictor)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
def root(body: RequestModel) -> ResponseModel:
    dialogue = body.model_dump()["conversation"]
    if len(dialogue) < 2:
        raise HTTPException(status_code=400, detail="Dialogue must contain at least 2 sentences.")
    label = predictor.predict([dialogue])[0]
    word_explanation = explainer.explain(dialogue, mode=0)
    sequence_explanation = explainer.explain(dialogue, mode=1)
    sentence_explanation = explainer.explain(dialogue, mode=2)
    return ResponseModel(
        label=label,
        word_explanation=word_explanation,
        sequence_explanation=sequence_explanation,
        sentence_explanation=sentence_explanation,
    )
