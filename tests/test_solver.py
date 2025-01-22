import json
import os
from types import SimpleNamespace

import pytest
import torch

from emsa.model import EpidemicModel
from emsa.utils import PROJECT_PATH
from emsa_examples.utils.dataloader_16_ag import DataLoader
from tests.mock_models import (
    MockModelBase,
    MockSEIRModel,
    MockSEIHRModel,
    MockContactModel,
    MockVaccinatedModel,
    EMSAMockVaccinatedModel,
)

# Constants for file paths
SEIR_CONFIG_PATH = "emsa_examples/SEIR_no_age_groups/configs/model_struct.json"
SEIHR_CONFIG_PATH = "emsa_examples/SEIHR_2_age_groups/configs/model_struct.json"
CONTACT_CONFIG_PATH = "emsa_examples/contact_sensitivity/configs/model_struct.json"
VACCINATED_CONFIG_PATH = "emsa_examples/vaccinated_sensitivity/configs/model_struct.json"


def load_model_struct(rel_path):
    """Load model structure from a JSON file."""
    with open(os.path.join(PROJECT_PATH, rel_path)) as f:
        model_struct = json.load(f)
    return model_struct


@pytest.fixture(scope="module")
def model_structs():
    """Load all model structures for the tests."""
    return {
        "seir": load_model_struct(SEIR_CONFIG_PATH),
        "seihr": load_model_struct(SEIHR_CONFIG_PATH),
        "contact": load_model_struct(CONTACT_CONFIG_PATH),
        "vacc": load_model_struct(VACCINATED_CONFIG_PATH),
    }


@pytest.fixture
def seir_data():
    """Fixture for SEIR model data."""
    data = DataLoader()
    contact_data_seir = torch.tensor(1)
    age_data_seir = torch.tensor([10000])
    return SimpleNamespace(
        params=data.params,
        cm=contact_data_seir,
        age_data=age_data_seir,
        n_age=1,
        device="cpu",
    )


@pytest.fixture
def seihr_data():
    """Fixture for SEIHR model data."""
    data = DataLoader()
    seihr_params = data.params.copy()
    seihr_params["eta"] = torch.tensor([0.1, 0.3])
    contact_data_seihr = torch.tensor([[1, 2], [0.5, 1]])
    age_data_seihr = torch.tensor([1e5, 2e5])
    return SimpleNamespace(
        params=seihr_params,
        cm=contact_data_seihr,
        age_data=age_data_seihr,
        n_age=2,
        device="cpu",
    )


@pytest.fixture
def contact_data():
    """Fixture for contact model data."""
    return DataLoader()


@pytest.fixture
def vaccinated_data():
    """Fixture for vaccinated model data."""
    return DataLoader()


@pytest.mark.parametrize(
    "model_cls, data_fixture, model_key",
    [
        (MockSEIRModel, "seir_data", "seir"),
        (MockSEIHRModel, "seihr_data", "seihr"),
        (MockContactModel, "contact_data", "contact"),
        (MockVaccinatedModel, "vaccinated_data", "vacc"),
    ],
)
def test_models(model_cls, data_fixture, model_key, request, model_structs):
    """Parameterized test for different models."""
    data = request.getfixturevalue(data_fixture)
    _test_model(model_cls(data=data), model_structs[model_key])


def _test_model(mock_model: MockModelBase, model_struct):
    """Helper function to test a model."""
    is_vacc = isinstance(mock_model, MockVaccinatedModel)
    if is_vacc:
        model = EMSAMockVaccinatedModel(data=mock_model.data, model_struct=model_struct)
    else:
        model = EpidemicModel(data=mock_model.data, model_struct=model_struct)

    exposed_iv = torch.zeros(model.n_age)
    exposed_iv[0] = 10
    iv = {"e": exposed_iv} if "l" not in model.state_data else {"l": exposed_iv}
    y0 = torch.atleast_2d(model.get_initial_values_from_dict(iv))
    t_eval = torch.atleast_2d(torch.arange(0, 200, 1))
    model.initialize_matrices()
    emsa_sol = (
        model.get_solution(y0, t_eval).ys
        if not is_vacc
        else model.get_solution(y0, t_eval, odefun=model.get_vacc_ode()).ys
    )
    mock_sol = model.get_sol_from_ode(y0, t_eval, odefun=mock_model.odefun).ys
    assert (
        torch.abs(mock_sol[:, -1] - emsa_sol[:, -1]).sum().item() < 1
    ), "Model solutions do not match"


if __name__ == "__main__":
    pytest.main(["-v"])
