import os

from emsa.generics.generic_model import GenericModel
from emsa.generics.generic_sampler import GenericSampler
from emsa.utils.dataloader import PROJECT_PATH
from emsa.utils.simulation_base import SimulationBase


class SimulationSEIR(SimulationBase):
    def __init__(self, data):
        model_struct_path = os.path.join(
            PROJECT_PATH, "emsa_examples/SEIR_no_age_groups/configs/model_struct.json"
        )
        config_path = os.path.join(
            PROJECT_PATH,
            "emsa_examples/SEIR_no_age_groups/configs/sampling_config.json",
        )
        super().__init__(
            data, model_struct_path=model_struct_path, sampling_config_path=config_path
        )
        self.folder_name = os.path.join(self.folder_name, "sens_data_SEIR_no_ag")

        # Initalize model
        self.model = GenericModel(sim_object=self)

    def run_sampling(self):
        for variable_params in self.variable_param_combinations:
            base_r0 = variable_params["r0"]
            beta = self.model.get_beta_from_r0(base_r0=base_r0, susceptibles=self.susceptibles)
            self.params["beta"] = beta

            param_generator = GenericSampler(sim_object=self, variable_params=variable_params)
            param_generator.run()
