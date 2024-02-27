import numpy as np

from src.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_state: dict, sim_obj):
        super().__init__(sim_state, sim_obj)
        self.sim_obj = sim_obj
        self.susc = sim_state["susc"]
        self.base_r0 = sim_state["base_r0"]
        self.target_var = sim_state["target_var"]

        self.lhs_boundaries = {
            "lower": np.full(fill_value=0.1, shape=self.sim_obj.upper_tri_size),
            "upper": np.ones(self.sim_obj.upper_tri_size)
                               }

    def run_sampling(self):
        lhs_table = self._get_lhs_table()
        self._get_sim_output(lhs_table)

    def _get_variable_parameters(self):
        return f'{self.susc}-{self.base_r0}-{self.target_var}'