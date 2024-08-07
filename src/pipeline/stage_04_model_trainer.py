import sys
from src.exceptions import CustomException
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src import logger


STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training_config = ModelTrainer(config=model_training_config)
        model_training_config.train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        model_training = ModelTrainingPipeline()
        model_training.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
