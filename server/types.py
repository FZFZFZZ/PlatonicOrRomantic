from pydantic import BaseModel

from models.predictor import TextData
from preprocess import Label

class RequestModel(BaseModel):
    conversation: list[TextData]

class ResponseModel(BaseModel):
    label: Label
    word_explanation: list[tuple[str, float]]
    sequence_explanation: list[Label]
    sentence_explanation: list[tuple[TextData, float]]
