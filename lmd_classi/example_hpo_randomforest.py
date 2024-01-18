# pip install setfit

# wandb login, logging enabled by default in SetFit

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datasets import Dataset, DatasetDict, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sentence_transformers import SentenceTransformer
from optuna import Trial


def main():
    # Load data
    filepath = "/kaggle/input/lmd-ukraine-annotated-v3/lmd_ukraine_annotated_V3.parquet"
    data = pd.read_parquet(filepath)

    # Data Preprocessing
    with_labels = data.query("label_text.notnull()")
    test_df = data.query("label_text.isnull()")
    train_df, eval_df = train_test_split(
        with_labels, test_size=0.3, stratify=with_labels["label_text"], random_state=40
    )

    label_mapping = {"pro_ukraine": 0, "pro_russia": 1, "other": 2}
    for df in [train_df, eval_df]:
        df["label"] = df["label_text"].map(label_mapping)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
    )

    num_classes = len(train_dataset.unique("label"))

    # Model initialization function
    def model_init(params):
        params = params or {}
        n_estimators = params.get("n_estimators", 100)
        max_depth = params.get("max_depth", None)
        model_body = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        model_head = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators
        )
        return SetFitModel(model_body=model_body, model_head=model_head)

    # Hyperparameter space definition
    def hp_space(trial):
        return {
            "seed": trial.suggest_int("seed", 1, 40),
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 500, 1000]),
            "max_depth": trial.suggest_categorical("max_depth", [2, 3, 5]),
        }

    # Training Arguments
    args = TrainingArguments(
        batch_size=32,
        body_learning_rate=3e-7,
        num_epochs=2,
        sampling_strategy="oversampling",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to=None,  # e.g wandb
        run_name="setfit_semi_max_rf_head_no_sampling_v3",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        column_mapping={"comment": "text", "label": "label"},
    )

    # Run hyperparameter search
    best_run = trainer.hyperparameter_search(
        direction="maximize", hp_space=hp_space, n_trials=6
    )
    print(best_run)

    # Final model training
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()

    # Save and load model for predictions
    save_directory = "dir_path_here"
    trainer.model._save_pretrained(save_directory=save_directory)
    model = SetFitModel.from_pretrained(save_directory)

    # Evaluate model
    metrics = trainer.evaluate()
    print(metrics)

    # Example predictions
    preds = model.predict(
        [
            "La Russie va gagner cette guerre, ils ont plus de ressources",
            "les journalistes sont corrompus, le traitement est partial",
            "les pauvres ukrainiens se font anéantir et subissent des crimes de guerre",
            "il faut fournir plus d'armes à l'ukraine",
        ]
    )
    print(preds)


if __name__ == "__main__":
    main()
