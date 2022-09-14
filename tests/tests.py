import numpy as np
from functools import partial
from sksurv.linear_model import CoxPHSurvivalAnalysis
from survLime import survlime_explainer
from survLime.datasets.load_datasets import Loader
from typing import List


def test_shape_veterans_preprocessed() -> None:
    loader = Loader(dataset_name="veterans")
    x, _, _ = loader.load_data()
    assert x.shape == (137, 6)


def test_shape_udca_preprocessed() -> None:
    loader = Loader(dataset_name="udca")
    x, _, _ = loader.load_data()
    assert x.shape == (170, 4)


def test_shape_lung_preprocessed() -> None:
    loader = Loader(dataset_name="lung")
    x, _, _ = loader.load_data()
    assert x.shape == (228, 8)


def test_shape_pbc_preprocessed() -> None:
    loader = Loader(dataset_name="pbc")
    x, _, _ = loader.load_data()
    assert x.shape == (419, 17)


def test_shape_vetearns_computed_weights() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 9


def test_shape_udca_computed_weights() -> None:
    loader = Loader(dataset_name="udca")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 4


def test_shape_lung_computed_weights() -> None:
    loader = Loader(dataset_name="lung")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 11


def test_shape_pbc_computed_weights() -> None:
    loader = Loader(dataset_name="pbc")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 22


def test_norm_less_than_one() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    try:
        _ = compute_weights(train, test, norm=0.5)
    except ValueError:
        pass


def compute_weights(train: np.array, test: np.array, norm: float = 2) -> List[float]:
    model = CoxPHSurvivalAnalysis(alpha=0.0001)

    model.fit(train[0], train[1])

    times_to_fill = list(set([x[1] for x in train[1]]))
    times_to_fill.sort()

    explainer = survlime_explainer.SurvLimeExplainer(
        train[0], train[1], model_output_times=model.event_times_
    )

    num_pat = 1000
    test_point = test[0].iloc[0]
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    b, _ = explainer.explain_instance(
        test_point,
        predict_chf,
        verbose=False,
        num_samples=num_pat,
        norm=norm,
    )
    b = [x[0] for x in b]
    return b
