import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import matplotlib.pyplot as plt

from typing import Any, Dict, Optional, Tuple

from collections import Counter
import pandas as pd


class DatasetManager:
    """
    Менеджер для загрузки, анализа и подготовки датасета для задачи детекции спама.
    """

    def __init__(
        self,
        dataset_name: str = "sms_spam",
        text_column: str = "sms",
        validation_split: float = 0.2,
        test_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Менеджер для загрузки, анализа и подготовки датасета для задачи детекции спама.

        Параметры:
            dataset_name (str): Название датасета.
            text_column (str): Колонка с текстом.
            validation_split (float): Доля валидационной выборки.
            test_split (float): Доля тестовой выборки.
            seed (int): Сид для воспроизводимости.
        """
        self.text_column: str = text_column
        raw = load_dataset(dataset_name)

        if "test" not in raw:
            train_temp = raw["train"].train_test_split(
                test_size=test_split + validation_split, seed=seed
            )

            temp = train_temp["test"].train_test_split(
                test_size=test_split / (test_split + validation_split), seed=seed
            )

            self.dataset = DatasetDict(
                {
                    "train": train_temp["train"],
                    "validation": temp["train"],
                    "test": temp["test"],
                }
            )
        else:
            self.dataset = raw

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def summary(self) -> None:
        """
        Выводит статистику по датасету:
        1. Распределение примеров между train/validation/test
        2. Распределение классов в обучающей выборке
        """
        sizes = {split: len(ds) for split, ds in self.dataset.items()}
        labels = list(self.dataset["train"]["label"])
        dist = Counter(labels)

        df1 = pd.DataFrame.from_dict(sizes, orient="index", columns=["Samples"])
        df2 = pd.DataFrame({"Label": list(dist.keys()), "Count": list(dist.values())})

        print("\033[92mDataset Splits:\033[0m")
        print(df1.to_markdown())
        print("\n\033[92mClass Distribution (train):\033[0m")
        print(df2.to_markdown(index=False))

    def show_samples(self, n: int = 5) -> None:
        """
        Отображает первые n записей с текстом и меткой.

        Параметры:
            n (int): число примеров.
        """
        ds = self.dataset["train"]
        print(f"\033[92mFirst {min(n, len(ds))} samples:\033[0m")
        for idx in range(min(n, len(ds))):
            example = ds[idx]
            label = example["label"]
            text = example[self.text_column]
            print(f"Sample {idx+1:2d}: [{label}] {text}")

    def preprocess(
        self,
        max_length: int = 128,
    ) -> DatasetDict:
        """
        Токенизация текстов и добавление поля 'labels'.

        Параметры:
            max_length (int): максимальная длина последовательности.

        Возвращает:
            DatasetDict: токенизированный датасет с 'input_ids', 'attention_mask', 'labels'.
        """

        def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:
            tokens = self.tokenizer(
                example[self.text_column],
                truncation=True,
                max_length=max_length,
            )
            tokens["labels"] = example["label"]
            return tokens

        tokenized = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        return tokenized


class SpamClassifier:
    """
    Обёртка для DistilBERT, обучаемая на задаче классификации спама.
    """

    def __init__(
        self,
        num_labels: int,
        output_dir: str = "./spam_model",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Инициализация модели и параметров тренировки.

        Параметры:
            num_labels (int): количество классов.
            output_dir (str): директория для результатов.
            epochs (int): число эпох.
            batch_size (int): размер батча.
            learning_rate (float): скорость обучения.
        """
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        self.args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_dir=f"{output_dir}/logs",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        self.trainer: Optional[Trainer] = None

    def compute_metrics(
        self,
        eval_pred: Tuple[Any, Any],
    ) -> Dict[str, float]:
        """
        Вычисляет accuracy по предсказаниям.

        Параметры:
            eval_pred: кортеж (логиты, метки).
        Возвращает:
            Dict[str, float]: {'accuracy': value}.
        """
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        accuracy = (preds == labels).astype(float).mean().item()
        return {"accuracy": accuracy}

    def train(
        self,
        tokenized_dataset: DatasetDict,
        tokenizer: AutoTokenizer,
    ) -> Trainer:
        """
        Запускает тренировку и валидацию модели.

        Параметры:
            tokenized_dataset: токенизированные 'train' и 'validation'.
            tokenizer: токенизатор для паддинга.
        Возвращает:
            Trainer: объект Trainer.
        """
        self.tokenizer = tokenizer
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        trainer.save_model(self.args.output_dir)
        self.trainer = trainer
        return trainer

    def test(self, tokenized_dataset: DatasetDict) -> Dict[str, float]:
        """
        Прогоняет модель на тестовом наборе и возвращает метрики.

        Параметры:
            tokenized_dataset (DatasetDict): Датасет с токенизированным сплитом 'test'.

        Возвращает:
            Dict[str, float]: Метрики оценки.
        """
        print("\n\033[92mTesting on test split...\033[0m")
        metrics = self.trainer.evaluate(eval_dataset=tokenized_dataset["test"])
        test_accuracy = metrics.get("eval_accuracy", 0.0)
        print(f"\033[92mTest Accuracy: {test_accuracy:.4f}\033[0m")
        return metrics

    def plot_loss(self) -> None:
        """
        Строит график train_loss, eval_loss и test_loss по эпохам.
        """
        train_epochs, train_losses = [], []
        eval_epochs, eval_losses = [], []
        for record in self.trainer.state.log_history:
            if "epoch" in record:
                if (
                    "loss" in record
                    and "eval_loss" not in record
                    and "test_loss" not in record
                ):
                    train_epochs.append(record["epoch"])
                    train_losses.append(record["loss"])
                if "eval_loss" in record:
                    eval_epochs.append(record["epoch"])
                    eval_losses.append(record["eval_loss"])
        plt.figure(figsize=(6, 4))
        plt.plot(train_epochs, train_losses, marker="o", label="Train Loss")
        plt.plot(eval_epochs, eval_losses, marker="o", label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_accuracy(self) -> None:
        """
        Строит график train_accuracy, eval_accuracy и test_accuracy по эпохам.
        """
        eval_epochs, eval_accs = [], []
        for record in self.trainer.state.log_history:
            if "epoch" in record:
                if "eval_accuracy" in record:
                    eval_epochs.append(record["epoch"])
                    eval_accs.append(record["eval_accuracy"])
        plt.figure(figsize=(6, 4))
        plt.plot(eval_epochs, eval_accs, marker="o", label="Eval Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


manager = DatasetManager(
    dataset_name="sms_spam",
    text_column="sms",
    validation_split=0.2,
    test_split=0.1,
    seed=42,
)
manager.summary()
manager.show_samples(n=10)
tokenized = manager.preprocess(max_length=128)

classifier = SpamClassifier(num_labels=2)

dummy_input_ids = torch.zeros((1, 128), dtype=torch.long)
dummy_attention = torch.ones((1, 128), dtype=torch.long)

trainer = classifier.train(
    tokenized_dataset=tokenized,
    tokenizer=manager.tokenizer,
)

classifier.plot_loss()
classifier.plot_accuracy()
classifier.test(tokenized)

print(
    "\033[92mTraining complete. Model and metrics saved to:\033[0m",
    classifier.args.output_dir,
)
