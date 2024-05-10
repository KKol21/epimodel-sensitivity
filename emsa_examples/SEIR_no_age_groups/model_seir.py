from emsa.sensitivity.sensitivity_model_base import SensitivityModelBase


class SEIRModel(SensitivityModelBase):
    def __init__(self, sim_object):
        super().__init__(sim_object=sim_object)

    def get_solution(self, y0, t_eval, **kwargs):
        odefun = self.get_basic_ode()
        return self.get_sol_from_ode(y0, t_eval, odefun)
