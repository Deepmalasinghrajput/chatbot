import sys
import os
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    # Use your trained model folder
    model_path = os.path.join("artifacts", "Chatbot_model")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def preprocess_function(self, examples, tokenizer):

        model_inputs = tokenizer(
            examples["query"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            text_target=examples["response"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded successfully")

            # ✅ Load tokenizer from LOCAL trained model
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path
            )

            # Convert pandas to HuggingFace Dataset
            train_data = Dataset.from_pandas(train_df)
            test_data = Dataset.from_pandas(test_df)

            # Apply preprocessing
            train_data = train_data.map(
                lambda x: self.preprocess_function(x, tokenizer),
                batched=True
            )

            test_data = test_data.map(
                lambda x: self.preprocess_function(x, tokenizer),
                batched=True
            )

            # Remove unwanted columns (important for Trainer)
            train_data = train_data.remove_columns(train_df.columns)
            test_data = test_data.remove_columns(test_df.columns)

            # Set format for PyTorch
            train_data.set_format(type="torch")
            test_data.set_format(type="torch")

            logging.info("Data transformation completed successfully")

            return train_data, test_data

        except Exception as e:
            raise CustomException(e, sys)