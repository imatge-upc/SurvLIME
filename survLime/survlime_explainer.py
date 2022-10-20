from functools import partial
from typing import Callable, Tuple, Union
import numpy as np
import cvxpy as cp
import sklearn
import sklearn.preprocessing
import pandas as pd
from sklearn.utils import check_random_state
from sksurv.util import Surv
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib
import plotly.subplots
import plotly.graph_objects
from survLime.utils.optimization import OptFuncionMaker
import shap


class SurvLimeExplainer:
    """To DO: change explanation
    Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        target_data: Union[np.ndarray, pd.DataFrame],
        model_output_times: np.ndarray,
        H0: np.ndarray = None,
        kernel_width: float = None,
        kernel: Callable = None,
        sample_around_instance: bool = False,
        random_state: int = None,
    ) -> None:

        """Init function.

        Args:
            training_data (Union[np.ndarray, pd.DataFrame]): data used to train the bb model
            target_data (Union[np.ndarray, pd.DataFrame]): target data used to train the bb model
            model_output_times (np.ndarray): output times of the bb model
            H0 (np.ndarray): baseline cumulative hazard
            kernel_width (float): width of the kernel to be used for computing distances
            kernel (Callable): kernel function to be used for computing distances
            sample_around_instance (bool): whether we sample around instances or not
            random_state (int): number to be used for random seeds

        Returns:
            None
        """

        self.training_data = training_data
        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.train_events = [y[0] for y in target_data]
        self.train_times = [y[1] for y in target_data]
        self.model_output_times = model_output_times
        if H0 is None:
            self.H0, self.timestamps = self.compute_nelson_aalen_estimator(
                self.train_events, self.train_times
            )
        else:
            self.H0 = H0

        # Validate H0 has the correct format
        # self.validate_H0(self.H0)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * 0.75
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d: np.ndarray, kernel_width: float) -> np.ndarray:
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)

        # Though set has no role to play if training data stats are provided
        # TO DO - Show Cris!
        # Instantiate an Scalar that will become important
        # take notice in the argument with_mean = False
        # We won't scale the data with the .transform method anyway
        # I tried switching it to false and it gave the same mean and variance
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)

    @staticmethod
    def compute_nelson_aalen_estimator(
        event: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        import ipdb;ipdb.set_trace()
        nelson_aalen = nelson_aalen_estimator(event, time)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        return H0, nelson_aalen[0]

    @staticmethod
    def validate_H0(H0: np.ndarray) -> None:
        if len(H0.shape) != 2:
            raise IndexError("H0 must be a 2 dimensional array.")
        if H0.shape[1] != 1:
            raise IndexError("The length of the last axis of must be equal to 1.")

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable,
        num_samples: int = 5000,
        distance_metric: str = "euclidean",
        norm: Union[float, str] = 2,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Generates explanations for a prediction.

        Args:
            data_row (np.ndarray): data point to be explained
            predict_fn (Callable): function that computes cumulative hazard
            num_samples (int): number of neighbours to use
            distance_metric (str): metric to be used for computing neighbours distance to the original point
            norm (Union[float, str]): number
            verbose (bool = False):

        Returns:
            b.values (np.ndarray): obtained weights from the convex problem.
            result (float): residual value of the convex problem.
        """

        scaled_data = self.generate_neighbours(data_row, num_samples)
        distances = sklearn.metrics.pairwise_distances(
            scaled_data, scaled_data[0].reshape(1, -1), metric=distance_metric  # TO DO
        ).ravel()

        weights = self.kernel_fn(distances)

        # Solution for the optimisation problems
        return self.solve_opt_problem(
            predict_fn=predict_fn,
            num_samples=num_samples,
            weights=weights,
            H0=self.H0,
            scaled_data=scaled_data,
            norm=norm,
            verbose=verbose,
        )

    def solve_opt_problem(
        self,
        predict_fn: Callable,
        num_samples: int,
        weights: np.ndarray,
        H0: np.ndarray,
        scaled_data: np.ndarray,
        norm: Union[float, str],
        verbose: float,
    ) -> Tuple[np.ndarray, float]:
        """Solves the convex problem proposed in: https://arxiv.org/pdf/2003.08371.pdfF

        Args:
            predict_fn (Callable): function to compute the cumulative hazard.
            num_samples (int): number of neighbours.
            weights (np.ndarray): distance weights computed for each data point.
            H0 (np.ndarray): baseline cumulative hazard.
            scaled_data (np.ndarray): original data point and the computed neighbours.
            norm (Union[float, str]: number of the norm to be computed in the cvx problem.
            verbose (float): activate verbosity of the cvxpy solver.

        Returns:
            b.values (np.ndarray): obtained weights from the convex problem.
            result (float): residual value of the convex problem.
        """
        epsilon = 0.00000001
        num_features = scaled_data.shape[1]
        m = len(set(self.train_times))
        # To do: validate H_i_j_wc
        H_i_j_wc = predict_fn(scaled_data)
        times_to_fill = list(set(self.train_times))
        times_to_fill.sort()
        H_i_j_wc = np.array(
            [
                np.interp(times_to_fill, self.model_output_times, H_i_j_wc[i])
                for i in range(H_i_j_wc.shape[0])
            ]
        )
        log_correction = np.divide(H_i_j_wc, np.log(H_i_j_wc + epsilon))

        # Varible to look for
        b = cp.Variable((num_features, 1))

        # Reshape and log of predictions
        H = np.reshape(np.array(H_i_j_wc), newshape=(num_samples, m))
        LnH = np.log(H + epsilon)

        # Log of baseline cumulative hazard
        LnH0 = np.log(H0 + epsilon)
        # Compute the log correction
        logs = np.reshape(log_correction, newshape=(num_samples, m))

        # Distance weights
        w = np.reshape(weights, newshape=(num_samples, 1))

        # Time differences
        t = self.train_times.copy()
        t.append(t[-1] + epsilon)
        t.sort()
        delta_t = [t[i + 1] - t[i] for i in range(m)]
        delta_t = np.reshape(np.array(delta_t), newshape=(m, 1))

        # Matrices to produce the proper sizes
        ones_N = np.ones(shape=(num_samples, 1))
        ones_m_1 = np.ones(shape=(m, 1))
        B = np.dot(ones_N, LnH0.T)
        C = LnH - B
        Z = scaled_data @ b
        D = Z @ ones_m_1.T
        E = C - D

        opt_maker = OptFuncionMaker(E, w, logs, delta_t)
        funct = opt_maker.compute_function(norm=norm)

        objective = cp.Minimize(funct)
        prob = cp.Problem(objective)
        result = prob.solve(verbose=verbose)

        labels = Surv().from_arrays(self.train_events, self.train_times)
        model = CoxPHSurvivalAnalysis().fit(self.training_data, labels)
        model.coef_ = b.value
        self.survlime_sf = model.predict_survival_function(
            scaled_data[0].reshape(1, -1), return_array=True
        )[0]
        self.predicted_sf = predict_fn(
            scaled_data[0].reshape(1, -1), return_array=True
        )[0]
        self.predicted_sf = np.exp(np.log(np.clip(1 - self.predicted_sf + epsilon,a_min=0, a_max=1)))
        self.predicted_sf =  np.interp(times_to_fill, self.model_output_times, self.predicted_sf)

        return b.value, result  # H_i_j_wc, weights, log_correction, scaled_data,

    def generate_neighbours(self, data_row: np.ndarray, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction.

        Args:
            data_row (np.ndarray): data point to be explained of shape (1 x features)
            num_samples (int): number of neighbours to generate

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features)
        """
        num_cols = data_row.shape[0]
        data = np.zeros((num_samples, num_cols))
        instance_sample = data_row
        scale = self.scaler.scale_
        mean = self.scaler.mean_
        data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(
            num_samples, num_cols
        )
        if self.sample_around_instance:
            data = data * scale + instance_sample
        else:
            data = data * scale + mean
        data[0] = data_row.copy()
        return data

    def plot(self, type="lines", show=True):
        ticker = matplotlib.ticker.MaxNLocator(
            nbins=10, min_n_ticks=4, integer=True, prune="upper"
        )
        ticks = ticker.tick_values(
            int(min(self.train_times)), int(max(self.train_times))
        ).astype(int)
        n_at_risk = []
        n_censored = []
        n_events = []
        for i in ticks:
            n_at_risk.append((self.train_times > i).sum())
            n_events.append(np.array(self.train_events)[self.train_times <= i].sum())
            n_censored.append(
                (~np.array(self.train_events)[self.train_times <= i]).sum()
            )
        if type == "lines":
            fig = plotly.subplots.make_subplots(
                rows=2, cols=1, print_grid=False, shared_xaxes=True
            )
            fig.add_trace(
                plotly.graph_objs.Scatter(
                    x=self.timestamps,
                    y=self.predicted_sf,
                    mode="lines",
                    line_color="#4378bf",
                    line_width=2,
                    hovertemplate="<b>Predicted SF</b><br>"
                    + "Time: %{x}<br>"
                    + "SF value: %{y:.6f}<extra></extra>",
                    hoverinfo="text",
                ),
                1,
                1,
            )
            fig.add_trace(
                plotly.graph_objs.Scatter(
                    x=self.timestamps,
                    y=self.survlime_sf,
                    mode="lines",
                    line_color="#371ea3",
                    line_width=2,
                    hovertemplate="<b>SurvLIME SF (LIME approx.)</b><br>"
                    + "Time: %{x}<br>"
                    + "SF value: %{y:.6f}<extra></extra>",
                    hoverinfo="text",
                ),
                1,
                1,
            )
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=ticks,
                    y=[0.8] * len(ticks),
                    text=n_at_risk,
                    mode="text",
                    showlegend=False,
                ),
                2,
                1,
            )
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=ticks,
                    y=[0.5] * len(ticks),
                    text=n_events,
                    mode="text",
                    showlegend=False,
                ),
                2,
                1,
            )
            fig.append_trace(
                plotly.graph_objs.Scatter(
                    x=ticks,
                    y=[0.2] * len(ticks),
                    text=n_censored,
                    mode="text",
                    showlegend=False,
                ),
                2,
                1,
            )
            x_range = [
                0 - int(max(self.train_times)) * 0.025,
                int(max(self.train_times)) * 1.025,
            ]
            fig.update_xaxes(
                {
                    "matches": None,
                    "showticklabels": True,
                    "title": "timeline",
                    "title_standoff": 0,
                    "type": "linear",
                    "gridwidth": 2,
                    "zeroline": False,
                    "automargin": True,
                    "tickmode": "array",
                    "tickvals": ticks,
                    "ticktext": ticks,
                    "tickcolor": "white",
                    "ticklen": 3,
                    "fixedrange": True,
                    "range": x_range,
                }
            ).update_yaxes(
                {
                    "type": "linear",
                    "gridwidth": 2,
                    "zeroline": False,
                    "automargin": True,
                    "ticks": "outside",
                    "tickcolor": "white",
                    "ticklen": 3,
                    "fixedrange": True,
                }
            ).update_layout(
                {
                    "showlegend": False,
                    "template": "none",
                    "margin_pad": 6,
                    "margin_l": 110,
                }
            ).update_layout(
                yaxis2={
                    "tickvals": [0.2, 0.5, 0.8],
                    "ticktext": ["censored", "events", "at risk"],
                }
            )
            fig["layout"]["xaxis2"]["visible"] = False
            fig["layout"]["yaxis2"]["showgrid"] = False
            fig["layout"]["yaxis"]["domain"] = [0.35, 1]
            fig["layout"]["yaxis2"]["domain"] = [0.0, 0.2]
            fig["layout"]["yaxis2"]["range"] = [0, 1]
        if show:
            fig.show(
                config={
                    "displaylogo": False,
                    "staticPlot": False,
                    "toImageButtonOptions": {
                        "height": None,
                        "width": None,
                    },
                    "modeBarButtonsToRemove": [
                        "sendDataToCloud",
                        "lasso2d",
                        "autoScale2d",
                        "select2d",
                        "zoom2d",
                        "pan2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "resetScale2d",
                        "toggleSpikelines",
                        "hoverCompareCartesian",
                        "hoverClosestCartesian",
                    ],
                }
            )
        else:
            return fig
