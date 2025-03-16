import argparse
import pandas as pd
from tabulate import tabulate

from .core import (
    TextPreprocessor,
    MLModels,
    DLModels,
    FineTuningModels,
    ModelEvaluator,
    load_dataset
)

from .utils import generate_report

def main():
    parser = argparse.ArgumentParser(description='FASTER CLI')
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--test', type=str, required=True, help='Path to test data')
    parser.add_argument('--report-path', type=str, default='./reports', help='Output report path')
    parser.add_argument('--report-name', type=str, default='report.pdf', help='Report name')
    parser.add_argument('--rows', type=int, help='Number of rows to process')
    
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_dataset(
        args.train, 
        args.test, 
        args.rows
    )
    
    # Initialize components
    preprocessor = TextPreprocessor()
    ml_models = MLModels()
    dl_models = DLModels()
    ft_models = FineTuningModels()
    evaluator = ModelEvaluator()
    
    results = []
    all_results = []

    for model_name, model in ml_models.models.items():
        for name, method in preprocessor.get_preprocessing_methods().items():
            processed_train_texts = X_train.apply(method)
            processed_test_texts = X_test.apply(method)
     
            result = evaluator.evaluate_machine_learning(
                model, model_name, processed_train_texts, y_train, processed_test_texts, y_test, name
            )
            results.append(result)
            result['Category'] = 'Classic Machine Learning'  
            all_results.append(result)
    
        results_df = pd.DataFrame(results)
        print(tabulate(results_df, headers='keys', tablefmt='grid'))
        print("\n\n")
        results = []

    for model_name, model in dl_models.models.items():
        for name, method in preprocessor.get_preprocessing_methods().items():
            processed_test_texts = X_test.apply(method)
            result = evaluator.evaluate_deep_learning(model, model_name, processed_test_texts, y_test, name)
            results.append(result)
            result['Category'] = 'Deep Learning (Pipeline)'  
            all_results.append(result)

        results_df = pd.DataFrame(results)
        print(tabulate(results_df, headers='keys', tablefmt='grid'))
        print("\n\n")
        results = []
    
    finetuned_results = []
    # Itera sui modelli e sui metodi di pre-processing
    for model_name, config in ft_models.models.items():
        for name, method in preprocessor.get_preprocessing_methods().items():
            # Se stai usando T5, converto le etichette numeriche in stringhe
            if "t5" in model_name.lower():
                train_labels_text = y_train.apply(lambda x: "positive" if x == 1 else "negative")
                test_labels_text = y_test.apply(lambda x: "positive" if x == 1 else "negative")
            else:
                train_labels_text = y_train
                test_labels_text = y_test

            result = evaluator.evaluate_deep_learning_finetuned(
                model_name=model_name,
                pretrained_model_name=config["pretrained_model_name"],
                tokenizer_name=config["tokenizer_name"],
                train_texts=X_train,
                train_labels=train_labels_text,
                test_texts=X_test,
                test_labels=test_labels_text,
                name_preprocessing=name,
                num_train_epochs=2
            )
            finetuned_results.append(result)
            result['Category'] = 'Deep Learning (Fine-Tuned)'  
            all_results.append(result)
        finetuned_results_df = pd.DataFrame(finetuned_results)
        print(tabulate(finetuned_results_df, headers='keys', tablefmt='grid'))
        print("\n\n")
        finetuned_results = []
    
    # Generate report
    generate_report(all_results, args.report_path, args.report_name)

if __name__ == "__main__":
    main()
