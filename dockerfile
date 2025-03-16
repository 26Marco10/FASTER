# Usa un'immagine Python ufficiale come base
FROM python:3.9-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia il contenuto della tua cartella locale nel container
COPY test.csv /app
COPY train.csv /app/
COPY test_twitter.csv /app
COPY train_twitter.csv /app/
COPY faster /app/faster
COPY setup.py /app

# Crea la cartella di output nel container
RUN mkdir -p /app/reports

# Installa le dipendenze di sistema per psutil (necessario per monitorare risorse)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev


# Aggiungi i comandi per eseguire il tuo programm
RUN pip install .
CMD ["faster", "--train", "train.csv", "--test", "test.csv", "--report-name", "report.pdf", "--rows", "2500"]

