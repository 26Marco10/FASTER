from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import pandas as pd


class MLModels:
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }

class DLModels:
    
    def __init__(self):
        self.models = {
            'RoBERTa': pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english"),
            'DistilBERT': pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"),
            'T5': pipeline("sentiment-analysis", model="google/flan-t5-large"),
            'Electra': pipeline("sentiment-analysis", model="google/electra-small-discriminator")
        }

class FineTuningModels:

    def __init__(self):
        self.models = {
            "RoBERTa": {
                "pretrained_model_name": "siebert/sentiment-roberta-large-english",
                "tokenizer_name": "siebert/sentiment-roberta-large-english"
            },
            "DistilBERT": {
                "pretrained_model_name": "distilbert-base-uncased-finetuned-sst-2-english",
                "tokenizer_name": "distilbert-base-uncased"
            },
            "T5": {
                "pretrained_model_name": "google-t5/t5-base",
                "tokenizer_name": "google-t5/t5-base"
            },
            "Electra": {
                "pretrained_model_name": "google/electra-small-discriminator",
                "tokenizer_name": "google/electra-small-discriminator"
            }
        }