import sys
from src.exceptions import CustomException
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src import logger


STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_dvc("train", model_evaluation_config.config.train_data_path)
        model_evaluation_config.log_into_dvc("test", model_evaluation_config.config.test_data_path)
        model_evaluation_config.live.make_summary()
        model_evaluation_config.plot_feature_importance()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise CustomException(e, sys)
