import numpy as np

from emsa.sensitivity.sampler_base import SamplerBase


class SamplerContact(SamplerBase):
    def __init__(self, sim_object, variable_params):
        super().__init__(sim_object, variable_params)

        self.lhs_bounds_dict = {
            "contacts": np.array(
                [
                    np.full(fill_value=0.2, shape=self.sim_object.upper_tri_size),
                    np.full(fill_value=1, shape=self.sim_object.upper_tri_size),
                ]
            ),
        }

    def run(self):
        lhs_table = self.get_lhs_table()
        self.get_sim_output(lhs_table=lhs_table)
