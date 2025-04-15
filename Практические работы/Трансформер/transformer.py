import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
from transformers import pipeline
from transformers import default_data_collator
from datasets import load_dataset


MODEL_NAME = "DeepPavlov/rubert-base-cased"


class QADatasetProcessor:
    """
    Класс для загрузки, токенизации и предобработки данных для задачи вопрос-ответ (QA).
    Использует SQuAD-подобный датасет SberQuad.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Инициализирует токенизатор на основе заданной модели.

        Параметры:
            model_name (str): название модели, совместимой с Hugging Face.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_data(self):
        """
        Загружает датасет SberQuad из Hugging Face Datasets.

        Возвращает:
            DatasetDict: словарь с разбивкой на train и validation.
        """
        self.dataset = load_dataset("kuznetsoffandrey/sberquad")
        return self.dataset

    def _preprocess_examples(self, examples):
        """
        Токенизирует и преобразует примеры в формат, совместимый с моделью вопрос-ответ.

        Параметры:
            examples (dict): батч примеров из датасета.

        Возвращает:
            dict: словарь с токенами и позициями начала и конца ответов.
        """
        questions = examples["question"]
        contexts = examples["context"]
        answers = examples["answers"]

        answer_starts = [ans["answer_start"][0] for ans in answers]
        answer_texts = [ans["text"][0] for ans in answers]

        inputs = self.tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        overflow_to_sample_mapping = inputs.pop("overflow_to_sample_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_idx = overflow_to_sample_mapping[i]
            start_char = answer_starts[sample_idx]
            end_char = start_char + len(answer_texts[sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            if not (
                offsets[context_start][0]
                <= start_char
                < end_char
                <= offsets[context_end][1]
            ):
                start_positions.append(0)
                end_positions.append(0)
                continue

            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= start_char:
                token_start += 1
            start_positions.append(token_start - 1)

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= end_char:
                token_end -= 1
            end_positions.append(token_end + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        inputs["overflow_to_sample_mapping"] = overflow_to_sample_mapping
        return inputs

    def get_tokenized_dataset(self, dataset):
        """
        Применяет токенизацию ко всему датасету.

        Параметры:
            dataset (DatasetDict): оригинальный датасет SberQuad.

        Возвращает:
            DatasetDict: токенизированный датасет.
        """
        return dataset.map(
            self._preprocess_examples,
            batched=True,
            remove_columns=dataset["train"].column_names,
            batch_size=100,
        )


class QAModelTrainer:
    """
    Класс для обучения модели на задаче вопрос-ответ с использованием Trainer API.
    """

    def __init__(
        self, model_name: str = MODEL_NAME, tokenizer=None, training_args: dict = None
    ):
        """
        Инициализация модели и аргументов тренировки.

        Параметры:
            model_name (str): имя модели.
            tokenizer: токенизатор, используемый в pipeline.
            training_args (dict): словарь с параметрами обучения.
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.training_args = training_args or {
            "output_dir": "./results",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 8,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_dir": "./logs",
            "fp16": torch.cuda.is_available(),
        }

    def setup_trainer(self, tokenized_dataset, original_dataset):
        """
        Создаёт Trainer с метриками, моделью и параметрами.

        Параметры:
            tokenized_dataset (DatasetDict): токенизированный датасет.
            original_dataset (DatasetDict): исходный SberQuad.

        Возвращает:
            Trainer: объект Trainer.
        """
        args = TrainingArguments(**self.training_args)
        squad_metric = evaluate.load("squad_v2")

        def compute_metrics(p):
            """Вычисляет метрики точности для задачи QA."""
            start_logits, end_logits = p.predictions
            start_pred = torch.argmax(torch.tensor(start_logits), dim=1).numpy()
            end_pred = torch.argmax(torch.tensor(end_logits), dim=1).numpy()

            formatted_predictions = []
            references = []

            for i in range(len(start_pred)):
                sample_idx = tokenized_dataset["validation"][i][
                    "overflow_to_sample_mapping"
                ]
                original_sample = original_dataset["validation"][sample_idx]

                prediction_text = self.tokenizer.decode(
                    tokenized_dataset["validation"][i]["input_ids"][
                        start_pred[i] : end_pred[i] + 1
                    ],
                    skip_special_tokens=True,
                )

                references.append(
                    {
                        "id": str(original_sample["id"]),
                        "answers": {
                            "text": original_sample["answers"]["text"],
                            "answer_start": original_sample["answers"]["answer_start"],
                        },
                    }
                )

                formatted_predictions.append(
                    {
                        "id": str(original_sample["id"]),
                        "prediction_text": prediction_text,
                        "no_answer_probability": 0.0,
                    }
                )

            return squad_metric.compute(
                predictions=formatted_predictions, references=references
            )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )
        return self.trainer

    def train(self):
        """Запуск обучения модели."""
        return self.trainer.train()

    def save_model(self, path: str = "./sberquad_qa_model"):
        """
        Сохраняет модель и токенизатор в указанную директорию.

        Параметры:
            path (str): путь для сохранения.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class QAPipeline:
    """
    Класс для выполнения предсказаний с помощью обученной модели в формате question-answering pipeline.
    """

    def __init__(self, model_path: str = "./sberquad_qa_model"):
        """
        Инициализирует модель и токенизатор из указанного пути.

        Параметры:
            model_path (str): путь до директории с моделью.
        """
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = pipeline(
            "question-answering",
            model = self.model,
            tokenizer = self.tokenizer,
            device = 0 if torch.cuda.is_available() else -1,
        )

    def predict(self, context: str, question: str):
        """
        Предсказывает ответ на вопрос по заданному контексту.

        Параметры:
            context (str): текстовый контекст.
            question (str): вопрос к контексту.

        Возвращает:
            dict: словарь с ответом и оценкой.
        """
        return self.pipeline(
            question=question,
            context=context,
            max_seq_len=384,
            doc_stride=128,
            handle_impossible_answer=True,
        )


if __name__ == "__main__":
    processor = QADatasetProcessor()
    dataset = processor.load_data()
    tokenized_dataset = processor.get_tokenized_dataset(dataset)

    trainer = QAModelTrainer(tokenizer=processor.tokenizer)
    trainer.setup_trainer(tokenized_dataset, dataset)
    trainer.train()
    trainer.save_model()

    qa_system = QAPipeline()
    context = (
        "Первые упоминания о строении человеческого тела встречаются в Древнем Египте."
    )
    question = "Где встречаются первые упоминания о строении человеческого тела?"

    result = qa_system.predict(context, question)
    print(f"Результирующий словарь: {result}")
    print(f"Результат предсказания:")
    print(f"Вопрос: {question}")
    print(f"Ответ: {result['answer']}")
    print(f"Точность: {result['score']:.2f}")

    print("📊 Результаты на тестовой выборке:")
    test_data = dataset["test"]

    for i, sample in enumerate(test_data.select(range(10))):
        context = sample["context"]
        question = sample["question"]
        true_answer = sample["answers"]["text"][0]

        prediction = qa_system.predict(context, question)

        print(f"Пример {i + 1}")
        print(f"Вопрос: {question}")
        print(f"Контекст: {context}")
        print(f"Правильный ответ: {true_answer}")
        print(f"Предсказание: {prediction['answer']}")
        print(f"Точность: {prediction['score']:.2f}")

    print("\n📊 Вычисляем метрики на всей тестовой выборке...")

    squad_metric = evaluate.load("squad_v2")

    test_data = dataset["test"]

    predictions = []
    references = []

    for sample in test_data:
        context = sample["context"]
        question = sample["question"]
        true_answers = sample["answers"]["text"]
        
        result = qa_system.predict(context, question)
        
        predictions.append({
            "id": str(sample["id"]),
            "prediction_text": result["answer"],
            "no_answer_probability": 0.0
        })
        
        references.append({
            "id": str(sample["id"]),
            "answers": {
                "text": true_answers,
                "answer_start": sample["answers"]["answer_start"]
            }
        })

    metrics = squad_metric.compute(predictions=predictions, references=references)

    print("📈 Метрики качества на тестовой выборке:")
    print(f"📌 Exact Match (EM): {metrics['exact']:.2f}")
    print(f"📌 F1 Score: {metrics['f1']:.2f}")
