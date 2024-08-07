import os
import numpy as np
from src import logger
from src.utils.common import load_data, save_data_to_csv
from sklearn.model_selection import train_test_split
from src.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    
    def train_test_spliting(self):
        data = load_data(self.config.data_path)
        # Drop rows with NaN values
        data = data.dropna()
        data['quality'] = np.where(data['quality']>6.5, 1,0)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(
            data,
            test_size=self.config.split_ratio,
            random_state=self.config.random_state)
        
        save_data_to_csv(train, self.config.train_file)
        save_data_to_csv(test, self.config.test_file)

        logger.info("Splited data into training and test sets")
        logger.info(f"Train file shape: {train.shape}")
        logger.info(f"Train file shape: {test.shape}")
        