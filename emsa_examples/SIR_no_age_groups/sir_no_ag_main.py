import os
from types import SimpleNamespace

import torch

from emsa.generics import SimulationGeneric
from emsa.utils.dataloader import PROJECT_PATH


def main():
    contact_data = torch.tensor(1)
    age_data = torch.tensor([10000])
    data = SimpleNamespace(
        **{
            "params": {},
            "cm": contact_data,
            "age_data": age_data,
            "n_age": 1,
            "device": "cpu",
        }
    )

    model_struct_path = os.path.join(
        PROJECT_PATH,
        "emsa_examples/SIR_no_age_groups/configs/model_struct.json",
    )
    sampling_config_path = os.path.join(
        PROJECT_PATH,
        "emsa_examples/SIR_no_age_groups/configs/sampling_config.json",
    )

    sim = SimulationGeneric(
        data=data,
        model_struct_path=model_struct_path,
        sampling_config_path=sampling_config_path,
        folder_name="sens_data_SIR_no_ag",
    )
    sim.run_sampling()
    sim.calculate_all_prcc()
    sim.calculate_all_p_values()
    sim.plot_all_prcc()


if __name__ == "__main__":
    main()
