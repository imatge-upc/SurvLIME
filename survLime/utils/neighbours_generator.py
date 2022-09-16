import numpy as np
import pandas as pd
from typing import List, Dict, Union


class NeighboursGenerator:
    """Manages the process of obtaining the Neighbours for a given row"""

    def __init__(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        data_row: np.ndarray,
        categorical_features: List[int] = None,
        random_state: int = None,
    ) -> None:

        """Init function.

        Args:
            training_data (Union[np.ndarray, pd.DataFrame]): data used to train the bb model
            data_row (np.ndarray): data point to be explained of shape (1 x features)
            categorical_features (List[int]): list of integeter indicating the categorical features
            random_state (int): number to be used for random seeds

        Returns:
            None
        """

        if isinstance(training_data, pd.DataFrame):
            self.training_data = training_data.to_numpy()
        else:
            self.training_data = training_data

        self.data_row = data_row
        self.total_features = training_data.shape[1]

        if categorical_features is None:
            self.cat_features = []
        else:
            self.cat_features = categorical_features

        self.cont_features = [
            i for i in range(self.total_features) if i not in categorical_features
        ]
        self.total_cat_features = len(self.cat_features)
        self.total_cont_features = len(self.cont_features)

        if len(self.cat_features) == 0:
            self.type_features = "continuous"
        elif len(self.cont_features) == 0:
            self.type_features = "categorical"
        else:
            self.type_features = "mixed"

        self.random_state = random_state

    @staticmethod
    def to_dict(keys, values):
        return {key: value for key, value in zip(keys, values)}

    def estimate_distribution_categorical_features(self) -> Dict:
        """Estimates the distribution for each categorical variable

        Returns:
            distribution (dict): a dictionary containing the distribution for each categorical variable
        """

        distribution = {}
        total = self.training_data.shape[0]
        if self.type_features in ["mixed", "categorical"]:
            for idx_feature in self.cat_features:
                feautre_values = self.training_data[:, idx_feature]
                unique, count = np.unique(feautre_values, return_counts=True)
                distribution[idx_feature] = self.to_dict(unique, count / total)
        return distribution

    def generate_cont_neighbours(
        self, num_samples: int, sample_around_instance: bool = False
    ) -> np.ndarray:
        """Generates a neighborhood around a prediction for continuous features

        Args:
            num_samples (int): number of neighbours to generate
            sample_around_instance (bool): whether we sample around instances or not

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features)
        """

        # Get continuous features
        training_data_cont = self.training_data[:, self.cont_features]

        # Estimate mean and variance for continuous features
        mean_value = np.nanmean(training_data_cont, axis=0, dtype=np.float64)
        sd_value = np.nanstd(training_data_cont, axis=0, dtype=np.float64)

        # Generate neighbours
        neighbours = self.random_state.normal(
            0, 1, size=(num_samples, self.total_cont_features)
        )
        neighbours *= sd_value

        if sample_around_instance:
            neighbours += self.data_row[self.cont_features]
        else:
            neighbours += mean_value

        return neighbours

    def generate_neighbours(
        self, num_samples: int, sample_around_instance: bool = False
    ) -> np.ndarray:
        """Generates a neighborhood around a prediction.

        Args:
            num_samples (int): number of neighbours to generate
            sample_around_instance (bool): whether we sample around instances or not

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features)
        """

        # 1 Generate neighbours for continuous features
        X_neigh_cont = self.generate_cont_neighbours(
            num_samples=num_samples, sample_around_instance=sample_around_instance
        )

        # 2 Generate neighbours for categorical features
        # 2.1 Generate distribution for categorical features
        # 2.2 Generate neighbours

        # 3 Merge both datasets
