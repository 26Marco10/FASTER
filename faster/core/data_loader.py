import pandas as pd

def load_dataset(train_path, test_path, num_rows=None):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if num_rows:
        train = train[:num_rows]
        test = test[:num_rows]
    
    return (
        train['text'], 
        train['label'],
        test['text'],
        test['label']
    )