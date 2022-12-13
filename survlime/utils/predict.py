import numpy as np
import pandas as pd
import copy
from sksurv.functions import StepFunction
from typing import Callable


def interpolate_values(
    matrix: np.array,
    unique_times_to_event: np.array,
    model_output_times: np.array,
) -> np.array:
    """Transform an array of step functions to a matrix

    Args:
        matrix (np.array): matrix with values.
        unique_times_to_event (np.array): unique times to event.
        model_output_times (np.array): times considered by teh model.

    Returns:
        interpolated_matrix (np.array): array containing the values.
    """
    interpolated_matrix = np.array(
        [
            np.interp(unique_times_to_event, model_output_times, matrix[i])
            for i in range(matrix.shape[0])
        ]
    )
    return interpolated_matrix


def validate_output_dimension(
    matrix: np.array, expected_num_rows: int, expected_num_cols: int
) -> None:
    """Validate the output

    Args:
        matrix (np.array): matrix to validate.
        expected_num_rows (int): expected number of rows.
        expected_num_cols (int): expected number of columns.
    """
    total_rows = matrix.shape[0]
    total_cols = matrix.shape[1]
    if total_rows != expected_num_rows or total_cols != expected_num_cols:
        raise ValueError(
            "Predicted function does not provide a prediction for all the individuals"
        )
    else:
        return None


def transform_step_function(array_step_functions: np.array) -> np.array:
    """Transform an array of step functions to a matrix

    Args:
        array_step_functions (np.array): array of step functions.

    Returns:
        matrix_values (np.array): array containing the values.
    """
    sample_value = array_step_functions[0]
    total_rows = len(array_step_functions)
    total_columns = len(sample_value.y)
    matrix_values = np.empty(shape=(total_rows, total_columns))
    for i, step_fn in enumerate(array_step_functions):
        matrix_values[i] = step_fn(step_fn.x)
    return matrix_values


def predict_wrapper(
    predict_fn: Callable,
    data: np.ndarray,
    unique_times_to_event: np.ndarray,
    model_output_times: np.ndarray,
) -> np.ndarray:
    """Return the matrix with the values predicted

    Args:
        predict_fn (Callable): function that computes cumulative hazard.
        data (np.ndarray): data to predict over.
        unique_times_to_event (np.ndarray): unique times to event.
        model_output_times (np.ndarray): output times of the bb model.

    Returns:
        predicted_values(np.ndarray): predicted values
    """
    num_individuals = data.shape[0]
    number_unique_times = unique_times_to_event.shape[0]
    # Predict
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy().astype(np.float32)
    
    values_raw = predict_fn(data)
    # In case of a pd.DataFrame, force numpy
    if isinstance(values_raw, pd.DataFrame):
        values = values_raw.to_numpy()
    else:
        values = copy.deepcopy(values_raw)

    # if the second dimension of values is larger, swap them
    if values.shape[1] > values.shape[0]:
        values = values.T
    # If it is a numpy array
    if isinstance(values, np.ndarray):
        total_dim = values.ndim
        # If it is a matrix
        if total_dim == 2:
            total_cols = values.shape[1]
            # The produced number of columns matches with the number of unique times to event
            if total_cols == number_unique_times:
                predicted_values = values
            # The number of columns does not match with the number of unique times to event
            else:
                predicted_values = interpolate_values(
                    matrix=values,
                    unique_times_to_event=unique_times_to_event,
                    model_output_times=model_output_times,
                )
        # It is an 1D array
        elif total_dim == 1:
            # It is a StepFunction
            if isinstance(values[0], StepFunction):
                predicted_values = transform_step_function(values)
                if predicted_values.shape[1] != number_unique_times:
                    predicted_values = interpolate_values(
                        matrix=predicted_values,
                        unique_times_to_event=unique_times_to_event,
                        model_output_times=model_output_times,
                    )
            else:
                raise NotImplemented("Unknown type of object")
    else:
        raise TypeError("Unknown type of object")

    validate_output_dimension(
        matrix=predicted_values,
        expected_num_rows=num_individuals,
        expected_num_cols=number_unique_times,
    )
    return predicted_values
