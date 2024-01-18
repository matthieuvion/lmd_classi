# pip install setfit

import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from transformers import EarlyStoppingCallback
from sentence_transformers.losses import CosineSimilarityLoss
from optuna import Trial


def main():
    # Load data
    filepath = "/kaggle/input/lmd-annotated/lmd_ukraine_annotated.parquet"
    data = pd.read_parquet(filepath)
    data["article_type"] = data["article_type"].astype(str)

    with_labels = data.query("classe.notnull()")
    test_df = data.query("classe.isnull()")

    train_df, eval_df = train_test_split(
        with_labels, test_size=0.4, stratify=with_labels["classe"], random_state=40
    )

    label_mapping = {"pro_ukraine": 0, "pro_russia": 1, "other": 2}
    for df in [train_df, eval_df]:
        df["label"] = df["classe"].map(label_mapping)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": eval_dataset, "test": test_dataset}
    )

    num_classes = len(train_dataset.unique("label"))

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        use_differentiable_head=True,
        head_params={"out_features": num_classes},
    )

    model.labels = ["pro_ukraine", "pro_russia", "other"]

    train_dataset = sample_dataset(
        dataset["train"], label_column="label", num_samples=72, seed=40
    )

    args = TrainingArguments(
        batch_size=(32, 16),
        num_epochs=(1, 16),
        end_to_end=True,
        body_learning_rate=(1e-07, 3e-06),
        head_learning_rate=2e-3,
        sampling_strategy="oversampling",
        max_steps=-1,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        metric="accuracy",
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        column_mapping={"comment": "text", "label": "label"},
    )

    trainer.train()
    trainer.evaluate()


if __name__ == "__main__":
    main()
s
