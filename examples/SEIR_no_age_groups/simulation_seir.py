import os

import torch

from examples.SEIR_no_age_groups.model_seir import SEIRModel
from examples.SEIR_no_age_groups.sampler_seir import SamplerSEIR
from src.dataloader import PROJECT_PATH
from src.simulation_base import SimulationBase


class SimulationSEIR(SimulationBase):
    def __init__(self, data):
        model_struct_path = os.path.join(PROJECT_PATH, "examples/SEIR_no_age_groups/configs/model_struct.json")
        config_path = os.path.join(PROJECT_PATH, "examples/SEIR_no_age_groups/configs/sampling_config.json")
        super().__init__(data, model_struct_path=model_struct_path, config_path=config_path)
        self.folder_name = os.path.join(self.folder_name, "sens_data_SEIR_no_ag")

        # Initalize model
        self.model = SEIRModel(sim_obj=self)

    def run_sampling(self):
        for variable_params in self.variable_param_combinations:
            susc = self.tensorize(param=variable_params["susc"])
            self.params.update({"susc": susc})
            base_r0 = variable_params["r0"]
            beta = self.get_beta_from_r0(base_r0)
            self.params["beta"] = beta

            self.model.initialize_matrices()

            param_generator = SamplerSEIR(sim_obj=self, variable_params=variable_params)
            param_generator.run()

    def tensorize(self, param):
        if isinstance(param, dict):
            return torch.Tensor(list(param.values())[0], device=self.device)
        else:
            return torch.Tensor(param, device=self.device)
