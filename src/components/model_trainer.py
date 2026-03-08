import os
import sys
from dataclasses import dataclass
import torch
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    # Where newly trained model will be saved
    trained_model_path = os.path.join("artifacts", "t5_model")

    # Path of your already trained Colab model
    base_model_path = os.path.join("artifacts", "Chatbot_model")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):

        try:
            logging.info("Loading LOCAL T5 model")

            # Load from your local trained model folder
            model = T5ForConditionalGeneration.from_pretrained(
                self.config.base_model_path
            )

            training_args = TrainingArguments(
                output_dir=self.config.trained_model_path,
                per_device_train_batch_size=4, 
                per_device_eval_batch_size=4,
                num_train_epochs=3,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_steps=10,
                learning_rate=3e-5,
                weight_decay=0.01,
                load_best_model_at_end=True,
                save_total_limit=1
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=test_data
            )

            logging.info("Starting model training")

            trainer.train()

            # Save final trained model
            model.save_pretrained(self.config.trained_model_path)

            logging.info("Model training completed and saved successfully")

        except Exception as e:
            raise CustomException(e, sys)