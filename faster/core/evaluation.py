import time
import tracemalloc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..utils.monitor import CpuMonitor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
import numpy as np
from .preprocessing import TextPreprocessor 
import evaluate


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        # Se i dati sono in formato pandas Series li trasformiamo in lista
        self.texts = texts.tolist() if isinstance(texts, pd.Series) else texts
        self.labels = labels.tolist() if isinstance(labels, pd.Series) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenizziamo il testo in input
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Rimuoviamo la dimensione extra introdotta da return_tensors
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Se la label è una stringa, la tokenizziamo
        if isinstance(self.labels[idx], str):
            label_encoding = self.tokenizer(
                self.labels[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )
            item['labels'] = label_encoding['input_ids'].squeeze(0)
        else:
            # Se invece è già numerica, la convertiamo in tensore
            item['labels'] = torch.tensor(self.labels[idx])
        return item

class ModelEvaluator:
    
    def __init__(self):
        pass
    
    @staticmethod
    def evaluate_classic_model(model, model_name, train_texts, train_labels, test_texts, test_labels, name_preprocessing):
        if "TF-IDF" in name_preprocessing:
            vectorizer = TfidfVectorizer()
        else:
            vectorizer = CountVectorizer()

        # Trasformazione dei dati
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        # Addestramento del modello
        model.fit(X_train, train_labels)

        # Inizia il monitoraggio
        monitor = CpuMonitor()

        start_time = time.time()
        tracemalloc.start()  # Avvia il monitoraggio della memoria
        monitor.start_cpu_monitoring()  # Avvia il monitoraggio della CPU
        # Predizione
        predictions = model.predict(X_test)

        cpu_usage = monitor.stop_cpu_monitoring()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        
        # Calcolo metriche
        execution_time = end_time - start_time
        memory_used_during_test = peak_memory / (1024 ** 2)  # Converti in MB

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        results = {
            "Model": f"{model_name} ({name_preprocessing})",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "Execution_time": execution_time,
            "Memory_usage_mb_during_test": memory_used_during_test,
            "Cpu_percent_during_test": cpu_usage,
        }

        return results
    
    @staticmethod
    def evaluate_deep_learning(model, model_name, test_texts, test_labels, name_preprocessing):
        # Inizializza la pipeline di sentiment analysis con RoBERTa
        sentiment_pipeline = model

        # Inizia il monitoraggio
        monitor = CpuMonitor()
        start_time = time.time()
        tracemalloc.start()  # Avvia il monitoraggio della memoria
        monitor.start_cpu_monitoring()  # Avvia il monitoraggio della CPU

        # Predizione
        predictions = [sentiment_pipeline(text, truncation=True)[0]['label'] for text in test_texts]

        cpu_usage = monitor.stop_cpu_monitoring()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()

        # Converte le predizioni in etichette numeriche (assumendo "POSITIVE" = 1, "NEGATIVE" = 0)
        predictions = [1 if label in {"LABEL_0", "POSITIVE"} else 0 for label in predictions]

        # Calcolo metriche
        execution_time = end_time - start_time
        memory_used_during_test = peak_memory / (1024 ** 2)  # Converti in MB

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)

        results = {
            "Model": f"{model_name} ({name_preprocessing})",
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "Execution_time": execution_time,
            "Memory_usage_mb_during_test": memory_used_during_test,
            "Cpu_percent_during_test": cpu_usage,
        }

        return results
    
    def evaluate_deep_learning_finetuned(self, model_name, pretrained_model_name, tokenizer_name, train_texts, train_labels, test_texts, test_labels, name_preprocessing, num_train_epochs=2):
        
        preprocessor = TextPreprocessor()
        method = preprocessor.get_preprocessing_methods().get(name_preprocessing)

        if not method:
            raise ValueError(f"Metodo di preprocessing non valido: {name_preprocessing}")

        processed_train_texts = train_texts.apply(method)
        processed_test_texts = test_texts.apply(method)
        
        if "t5" in model_name.lower():
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
            model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
        
        # Crea i dataset per Trainer
        train_dataset = CustomDataset(processed_train_texts, train_labels, tokenizer)
        test_dataset = CustomDataset(processed_test_texts, test_labels, tokenizer)
        
        # Definisci le metriche (ad es. accuracy, precision, recall, f1)
        metric_accuracy = evaluate.load("accuracy")
        metric_precision = evaluate.load("precision")
        metric_recall = evaluate.load("recall")
        metric_f1 = evaluate.load("f1")

        
        def compute_metrics(eval_pred):
            # Estrai logits e labels dalla tupla
            logits, labels = eval_pred

            # Se logits è una tupla, estrai il primo elemento (assumendo che siano i logits)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Controlla se stiamo usando T5 (output è una sequenza di token ID)
            if len(logits.shape) == 3:  # T5 restituisce (batch_size, sequence_length, vocab_size)
                # Decodifica i token ID in testo
                predictions = np.argmax(logits, axis=-1)
                decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
                binary_predictions = [1 if "positive" in pred.lower() else 0 for pred in decoded_predictions]

                # Decodifica le labels (se sono token ID)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                binary_labels = [1 if "positive" in label.lower() else 0 for label in decoded_labels]
            else:
                # Modelli di classificazione: logits sono direttamente interpretabili come probabilità
                predictions = np.argmax(logits, axis=-1)
                binary_predictions = predictions
                binary_labels = labels

            # Calcola le metriche
            accuracy = metric_accuracy.compute(predictions=binary_predictions, references=binary_labels)["accuracy"]
            precision = metric_precision.compute(predictions=binary_predictions, references=binary_labels, average="binary")["precision"]
            recall = metric_recall.compute(predictions=binary_predictions, references=binary_labels, average="binary")["recall"]
            f1 = metric_f1.compute(predictions=binary_predictions, references=binary_labels, average="binary")["f1"]

            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}     
        
        # Definisci i parametri di training
        training_args = TrainingArguments(
            output_dir=f"./results/{model_name}_{name_preprocessing}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            disable_tqdm=True,
            learning_rate=2e-5,
            weight_decay=0.01,
            seed=42
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Fine-tuning: addestra il modello
        trainer.train()
        
        # Monitoraggio: CPU, memoria e tempo
        monitor = CpuMonitor()
        start_time = time.time()
        tracemalloc.start()
        monitor.start_cpu_monitoring()

        # Valutazione
        eval_results = trainer.evaluate()
        
        cpu_usage = monitor.stop_cpu_monitoring()
        current, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        execution_time = end_time - start_time
        memory_used_during_test = peak_memory / (1024 ** 2)  # in MB
        
        # Costruisci un dizionario dei risultati
        results = {
            "Model": f"{model_name} fine-tuned ({name_preprocessing})",
            "Accuracy": eval_results.get("eval_accuracy", None),
            "Precision": eval_results.get("eval_precision", None),
            "Recall": eval_results.get("eval_recall", None),
            "F1_score": eval_results.get("eval_f1", None),
            "Execution_time": execution_time,
            "Memory_usage_mb_during_test": memory_used_during_test,
            "Cpu_percent_during_test": cpu_usage,
        }
        return results