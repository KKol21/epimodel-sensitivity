import os

import torch

from emsa.generics.generic_model import GenericModel
from emsa.generics.generic_sampler import GenericSampler
from emsa.utils.dataloader import PROJECT_PATH
from emsa.utils.simulation_base import SimulationBase


class SimulationSEIHR(SimulationBase):
    def __init__(self, data):
        model_struct_path = os.path.join(
            PROJECT_PATH, "emsa_examples/SEIHR_2_age_groups/configs/model_struct.json"
        )
        config_path = os.path.join(
            PROJECT_PATH,
            "emsa_examples/SEIHR_2_age_groups/configs/sampling_config.json",
        )
        super().__init__(
            data=data,
            model_struct_path=model_struct_path,
            sampling_config_path=config_path,
        )
        self.folder_name += "/sens_data_SEIR_2_ag"

        # Initalize model
        self.model = GenericModel(sim_object=self)

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_contact/lhs'
        and 'sens_data_contact/simulations' directories, respectively.

        """
        for variable_params in self.variable_param_combinations:
            susc = torch.Tensor(list(variable_params["susc"].values())[0], device=self.device)
            self.params.update({"susc": susc})
            base_r0 = variable_params["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params["beta"] = beta

            param_generator = GenericSampler(sim_object=self, variable_params=variable_params)
            param_generator.run()
