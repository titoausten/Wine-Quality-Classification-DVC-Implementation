from src import logger
from src.utils.common import load_data, save_bin
#from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from src.entity.config_entity import ModelTrainingConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    
    def train(self):
        # Load data
        train_data = load_data(self.config.train_data_path)
        test_data = load_data(self.config.test_data_path)

        # Features
        train_x = train_data.drop(self.config.target_column, axis=1)
        test_x = test_data.drop(self.config.target_column, axis=1)

        #Labels
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]


        #lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        #lr.fit(train_x, train_y)

        rfc = RandomForestClassifier(
            random_state = self.config.random_state,
            n_estimators= self.config.n_est)
        
        # Train model
        logger.info(f"Training started...")
        rfc.fit(train_x, train_y)
        logger.info(f"Training completed")


        # Save model
        #joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        save_bin(rfc, self.config.model_name)
