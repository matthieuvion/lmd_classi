import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

from sentence_transformers.losses import CosineSimilarityLoss
from optuna import Trial


def main():
    filepath = "/kaggle/input/lmd-ukraine-annotated-v3/lmd_ukraine_annotated_V3.parquet"

    data = pd.read_parquet(filepath)

    # Labeled data is split between train and eval sets
    # Test set will be the unlabeled data. Will be used for distillation in a later stage.
    with_labels = data.query("label_text.notnull()")
    test_df = data.query("label_text.isnull()")
    print(len(with_labels), len(test_df))

    # Labeled data split
    # Optional stratify= but we still want to make sure classes are "balanced" in both dataset
    train_df, eval_df = train_test_split(
        with_labels, test_size=0.3, stratify=with_labels["label_text"], random_state=40
    )

    label_mapping = {"pro_ukraine": 0, "pro_russia": 1, "other": 2}
    for df in [train_df, eval_df]:
        df["label"] = df["label_text"].map(label_mapping)

    # convert to HuggingFace DatasetDict format, for convenience
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
    )

    # Optional, save # classes. Used with a torch head.
    # num_classes = len(train_dataset.unique("label"))

    train_dataset = sample_dataset(
        dataset["train"], label_column="label", num_samples=80, seed=40
    )

    def model_init(params):
        params = params or {}
        max_iter = params.get("max_iter", 100)
        solver = params.get("solver", "liblinear")
        params = {
            "head_params": {
                "max_iter": max_iter,
                "solver": solver,
            }
        }

        return SetFitModel.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", **params
        )

    def hp_space(trial):
        """Define hyperparams search space (Optuna)"""

        return {
            # Embeddings fine-tuning phase params :
            "body_learning_rate": trial.suggest_float(
                "body_learning_rate", 1e-07, 3e-06, log=True
            ),
            "max_steps": trial.suggest_int("max_steps", 150, 600),
            "batch_size": trial.suggest_categorical("batch_size", [32]),
            "seed": trial.suggest_int("seed", 1, 40),
            # LogisticRegression head params :
            "max_iter": trial.suggest_int("max_iter", 120, 140),
            "solver": trial.suggest_categorical("solver", ["liblinear"]),
        }

    args = TrainingArguments(
        sampling_strategy="oversampling",
        evaluation_strategy="steps",
        eval_steps=20,  # print eval every eval_steps
        save_strategy="steps",
    )

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        column_mapping={"comment": "text", "label": "label"},
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize", hp_space=hp_space, n_trials=10
    )
    print(best_run)


if __name__ == "__main__":
    main()
