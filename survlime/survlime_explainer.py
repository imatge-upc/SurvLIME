from functools import partial
from copy import deepcopy
from typing import Callable, Tuple, Union, List, Literal
import numpy as np
import pandas as pd
import cvxpy as cp
import sklearn
import sklearn.preprocessing
import pandas as pd
from sklearn.utils import check_random_state
from sksurv.nonparametric import nelson_aalen_estimator
from survlime.utils.optimization import OptFuncionMaker
from survlime.utils.neighbours_generator import NeighboursGenerator
from survlime.utils.predict import predict_wrapper


class SurvLimeExplainer:
    """
    Look for the coefficient of a COX model."""

    def __init__(
        self,
        training_features: Union[np.ndarray, pd.DataFrame],
        traininig_events: List[Union[float, int]],
        training_times: List[Union[bool, float, int]],
        model_output_times: np.ndarray,
        categorical_features: List[int] = None,
        H0: np.ndarray = None,
        kernel_width: float = None,
        kernel: Callable = None,
        sample_around_instance: bool = False,
        random_state: int = None,
    ) -> None:
        """Init function.
        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model
            traininig_events (List[Union[float, int]]): training events indicator
            training_times (List[Union[bool, float, int]]): training times to event
            model_output_times (np.ndarray): output times of the bb model
            categorical_features (List[int]): list of integeter indicating the categorical features
            H0 (np.ndarray): baseline cumulative hazard
            kernel_width (float): width of the kernel to be used for computing distances
            kernel (Callable): kernel function to be used for computing distances
            sample_around_instance (bool): whether we sample around instances or not
            random_state (int): number to be used for random seeds
        Returns:
            None
        """

        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.training_data = training_features
        self.train_events = traininig_events
        self.train_times = training_times
        self.categorical_features = categorical_features
        self.model_output_times = model_output_times
        if H0 is None:
            self.H0 = self.compute_nelson_aalen_estimator(
                self.train_events, self.train_times
            )
        else:
            self.H0 = H0

        # Validate H0 has the correct format
        # self.validate_H0(self.H0)

        if kernel_width is None:
            kernel_width = np.sqrt(self.training_data.shape[1]) * 0.75
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d: np.ndarray, kernel_width: float) -> np.ndarray:
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(self.training_data)
        self.is_data_frame = isinstance(self.training_data, pd.DataFrame)
        if self.is_data_frame:
            self.column_names = self.training_data.columns
        else:
            self.column_names = None

    @staticmethod
    def compute_nelson_aalen_estimator(
        event: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        nelson_aalen = nelson_aalen_estimator(event, time)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        return H0

    @staticmethod
    def validate_H0(H0: np.ndarray) -> None:
        if len(H0.shape) != 2:
            raise IndexError("H0 must be a 2 dimensional array.")
        if H0.shape[1] != 1:
            raise IndexError("The length of the last axis of must be equal to 1.")

    def transform_data(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transforms data according with what the model needs
        Args:
            data (Union[np.ndarray, pd.DataFrame]): data to be transformed

        Returns:
            data_transformed (Union[np.ndarray, pd.DataFrame]): transformed data
        """
        is_orinial_data_pandas = isinstance(self.training_data, pd.DataFrame)
        if is_orinial_data_pandas:
            data_transformed = pd.DataFrame(data, columns=self.column_names)
        else:
            training_dtype = self.training_data.dtype
            data_transformed = np.copy(data)
            data_transformed = data_transformed.astype(training_dtype)
        return data_transformed

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"] = "cumulative",
        num_samples: int = 5000,
        distance_metric: str = "euclidean",
        norm: Union[float, str] = 2,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Generates explanations for a prediction.
        Args:
            data_row (np.ndarray): data point to be explained
            predict_fn (Callable): function that computes cumulative hazard
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function
            num_samples (int): number of neighbours to use
            distance_metric (str): metric to be used for computing neighbours distance to the original point
            norm (Union[float, str]): number
            verbose (bool = False):
        Returns:
            b.values (np.ndarray): obtained weights from the convex problem.
            result (float): residual value of the convex problem.
        """

        neighbours_generator = NeighboursGenerator(
            training_data=self.training_data,
            data_row=data_row,
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )

        scaled_data = neighbours_generator.generate_neighbours(
            num_samples=num_samples, sample_around_instance=self.sample_around_instance
        )

        scaled_data_transformed = self.transform_data(data=scaled_data)

        distances = sklearn.metrics.pairwise_distances(
            scaled_data, scaled_data[0].reshape(1, -1), metric=distance_metric
        ).ravel()

        weights = self.kernel_fn(distances)

        # Solution for the optimisation problems
        return self.solve_opt_problem(
            predict_fn=predict_fn,
            type_fn=type_fn,
            num_samples=num_samples,
            weights=weights,
            H0=self.H0,
            data=scaled_data_transformed,
            norm=norm,
            verbose=verbose,
        )

    def solve_opt_problem(
        self,
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"],
        num_samples: int,
        weights: np.ndarray,
        H0: np.ndarray,
        data: np.ndarray,
        norm: Union[float, str],
        verbose: bool,
    ) -> Tuple[np.ndarray, float]:
        """Solves the convex problem proposed in: https://arxiv.org/pdf/2003.08371.pdfF
        Args:
            predict_fn (Callable): function to compute the cumulative hazard
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function
            num_samples (int): number of neighbours
            weights (np.ndarray): distance weights computed for each data point
            H0 (np.ndarray): baseline cumulative hazard
            data (np.ndarray): original data point and the computed neighbours
            norm (Union[float, str]: number of the norm to be computed in the cvx problem
            verbose (bool): activate verbosity of the cvxpy solver

        Returns:
            cox_coefficients (np.ndarray): coefficients of a COX model
        """
        epsilon = 10 ** (-6)
        num_features = data.shape[1]
        unique_times_to_event = np.sort(np.unique(self.train_times))
        m = unique_times_to_event.shape[0]


        FN_pred = predict_wrapper(
            predict_fn=predict_fn,
            data=data,
            unique_times_to_event=unique_times_to_event,
            model_output_times=self.model_output_times,
        )
        # From now on, original data must be a numpy array
        if isinstance(data, np.ndarray):
            data_np = data
        elif isinstance(data, pd.DataFrame):
            data_np = data.to_numpy()
        else:
            raise TypeError(
                f"Unknown data type {type(data)} only np.ndarray or pd.DataFrame allowed"
            )

        # Varible to look for
        b = cp.Variable((num_features, 1))

        # Reshape and log of predictions
        if type_fn == "survival":
            H_score = -np.log(FN_pred + epsilon)
        elif type_fn == "cumulative":
            H_score = deepcopy(FN_pred)
        else:
            raise ValueError("type_fn must be either survival or cumulative string")
        log_correction = np.divide(H_score, np.log(H_score + epsilon))
        H = np.reshape(np.array(H_score), newshape=(num_samples, m))
        LnH = np.log(H + epsilon)

        # Log of baseline cumulative hazard
        LnH0 = np.log(H0 + epsilon)
        # Compute the log correction
        logs = np.reshape(log_correction, newshape=(num_samples, m))

        # Distance weights
        w = np.reshape(weights, newshape=(num_samples, 1))

        # Time differences
        t = np.empty(shape=(m + 1, 1))
        t[:m, 0] = unique_times_to_event
        t[m, 0] = t[m - 1, 0] + epsilon
        delta_t = [t[i + 1, 0] - t[i, 0] for i in range(m)]
        delta_t = np.reshape(np.array(delta_t), newshape=(m, 1))

        # Matrices to produce the proper sizes
        ones_N = np.ones(shape=(num_samples, 1))
        ones_m_1 = np.ones(shape=(m, 1))
        B = np.dot(ones_N, LnH0.T)
        C = LnH - B
        Z = data_np @ b
        D = Z @ ones_m_1.T
        E = C - D

        opt_maker = OptFuncionMaker(E, w, logs, delta_t)
        funct = opt_maker.compute_function(norm=norm)

        objective = cp.Minimize(funct)
        prob = cp.Problem(objective)


       #for solver in cp.installed_solvers():
       #    print(f'Trying solver {solver}')
      # try:
      #     verbose=True
        result = prob.solve(verbose=verbose)
       #result = prob.solve(verbose=verbose, solver='ECOS', abstol=1e-4)
      # except:
      #     print(f'Failed to solve with {solver}')
        cox_coefficients = b.value[:, 0]

        return cox_coefficients
