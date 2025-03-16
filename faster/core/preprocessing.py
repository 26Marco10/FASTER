import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import random
from sklearn.feature_extraction.text import TfidfVectorizer



# Scarica risorse di nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download("punkt_tab")

class TextPreprocessor:
    # Pre-processing base del testo
    @staticmethod
    def basic_preprocess_text(text):
        text = str(text)
        # Rimuovi tag HTML
        text = re.sub(r"<.*?>", " ", text)
        # Rimuovi caratteri speciali e numeri
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Trasforma in minuscolo
        text = text.lower()
        # Tokenizza il testo
        tokens = word_tokenize(text)
        # Rimuovi stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
        # Applica stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        # Unisci i token in una stringa
        return " ".join(tokens)

    @staticmethod
    def first_half(text):
        sentences = sent_tokenize(text)
        half = len(sentences) // 2  # Prendi la metà del numero di frasi
        half_sentences = " ".join(sentences[:half]) # Restituisce solo la prima metà del testo
        if TextPreprocessor.basic_preprocess_text(half_sentences) == "":
            return half_sentences
        return TextPreprocessor.basic_preprocess_text(half_sentences)

    @staticmethod
    def second_half(text):
        sentences = sent_tokenize(text)
        half = len(sentences) // 2  # Calcola la metà del numero di frasi
        half_sentences = " ".join(sentences[half:]) # Restituisce solo la seconda metà del testo
        if TextPreprocessor.basic_preprocess_text(half_sentences) == "":
            return half_sentences
        return TextPreprocessor.basic_preprocess_text(half_sentences)  
    
    @staticmethod
    def first_sentence(text):
        sentences = sent_tokenize(text)
        first_sentence = sentences[0] if sentences else ""  # Ottieni la prima frase, se disponibile
        if TextPreprocessor.basic_preprocess_text(first_sentence) == "":
            return first_sentence
        return TextPreprocessor.basic_preprocess_text(first_sentence)  
    
    @staticmethod
    def last_sentence(text):
        sentences = sent_tokenize(text)
        last_sentence = sentences[-1] if sentences else ""  # Ottieni l'ultima frase, se disponibile
        if TextPreprocessor.basic_preprocess_text(last_sentence) == "":
            return last_sentence
        return TextPreprocessor.basic_preprocess_text(last_sentence)  

    @staticmethod
    def random(text):
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        if num_sentences <= 1:  # Se c'è solo una frase, restituiscila
            return text
        # Determina un punto iniziale casuale
        start = random.randint(0, num_sentences - 1)
        # Determina una lunghezza casuale della porzione
        length = random.randint(1, num_sentences - start)
        random_text = " ".join(sentences[start:start + length])
        if TextPreprocessor.basic_preprocess_text(random_text) == "":
            return random_text
        return TextPreprocessor.basic_preprocess_text(random_text)

    @staticmethod
    def vader(text, threshold=0.5):
        # Pre-processing base del testo
        preprocessed_text = TextPreprocessor.basic_preprocess_text(text)
        # Inizializza VADER
        sia = SentimentIntensityAnalyzer()
        tokens = preprocessed_text.split()
        selected_words = []
        for word in tokens:
            # Ottieni il punteggio di sentiment di VADER per ogni parola
            sentiment_score = sia.polarity_scores(word)
            # Includi la parola se il punteggio positivo o negativo supera la soglia
            if sentiment_score["pos"] >= threshold or sentiment_score["neg"] >= threshold:
                selected_words.append(word)
        return " ".join(selected_words)  # Restituisce una stringa

    
    @staticmethod
    def td_idf_n_words(text, top_n=10):
        # Assicurati che text sia una stringa
        if not isinstance(text, str):
            text = str(text)
        
        # Applica il pre-processing di base
        preprocessed_text = TextPreprocessor.basic_preprocess_text(text)

        # Se il testo processato è vuoto, restituisci il testo originale o un fallback
        if not preprocessed_text.strip():
            return text  # oppure: return "nessuna_parola"

        # Usa un token_pattern che accetta anche parole di un solo carattere
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        
        try:
            tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        except ValueError as e:
            # Se per qualche motivo il vettorizzatore continua a dare errore, restituisci un fallback
            print("Errore nel vettorizzatore:", e, "per il testo:", preprocessed_text)
            return "nessuna_parola"
        
        feature_names = vectorizer.get_feature_names_out()
        
        # Ottieni le parole con il punteggio più alto
        row = tfidf_matrix.toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_indices if row[i] > 0]

        return " ".join(top_words)
    
    @staticmethod
    def get_preprocessing_methods():
        return {
            'Basic Preprocessing': TextPreprocessor.basic_preprocess_text,
            'First half': TextPreprocessor.first_half,
            'Second half': TextPreprocessor.second_half,
            'First sentence': TextPreprocessor.first_sentence,
            'Last sentence': TextPreprocessor.last_sentence,
            'Random text': TextPreprocessor.random,
            'VADER': TextPreprocessor.vader,
            'TF-IDF 10': lambda x: TextPreprocessor.td_idf_n_words(x, 10),
            'TF-IDF 20': lambda x: TextPreprocessor.td_idf_n_words(x, 20),
            'TF-IDF 30': lambda x: TextPreprocessor.td_idf_n_words(x, 30)
        }
