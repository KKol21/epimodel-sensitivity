from emsa.sensitivity import SamplerBase


class GenericSampler(SamplerBase):
    def __init__(self, sim_object, variable_params=None):
        super().__init__(sim_object, variable_params)

    def run(self):
        lhs_table = self.get_lhs_table()
        self.get_sim_output(lhs_table)
