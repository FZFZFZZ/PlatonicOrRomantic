from fastapi import FastAPI
import logging

from models.predictor import LstmPredictor
from models.explainer import LimeExplainer
from .types import RequestModel, ResponseModel

logging.basicConfig(level=logging.INFO)

app = FastAPI()

predictor = LstmPredictor()
explainer = LimeExplainer(predictor)

@app.post("/evaluate")
def root(body: RequestModel) -> ResponseModel:
    dialogue = body.model_dump()["conversation"]
    print(f"{dialogue=}")
    label = predictor.predict([dialogue])[0]
    word_explanation = explainer.explain(dialogue, mode=0)
    # sequence_explanation = explainer.explain(dialogue, mode=1)
    sentence_explanation = explainer.explain(dialogue, mode=2)
    return ResponseModel(
        label=label,
        word_explanation=word_explanation,
        sentence_explanation=sentence_explanation,
    )
