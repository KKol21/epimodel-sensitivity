import torch

from emsa.sensitivity.sensitivity_model_base import SensitivityModelBase


class VaccinatedModel(SensitivityModelBase):
    def __init__(self, sim_object):
        """
        Initializes the VaccinatedModel class.

        This method initializes the VaccinatedModel class by calling the parent class (EpidemicModelBase)
        constructor, and instantiating the matrix generator used in solving the model.

        Args:
            sim_object (SimulationVaccinated): Simulation object

        """
        super().__init__(sim_object=sim_object)

        self.V_1 = None
        self.V_2 = None

    def get_solution(self, y0, t_eval, **kwargs):
        lhs_table = kwargs["lhs_table"]
        self.initialize_matrices()
        self.V_1 = self._get_V_1_from_lhs(lhs_table=lhs_table)
        self.V_2 = self.matrix_generator.get_V_2()
        odefun = self.get_vaccinated_ode(curr_batch_size=lhs_table.shape[0])
        return self.get_sol_from_ode(y0, t_eval, odefun)

    def get_vaccinated_ode(self, curr_batch_size):
        V_1_mul = self.get_mul_method(self.V_1)

        v_div = torch.ones((curr_batch_size, self.n_eq)).to(self.device)
        div_idx = self.idx("s_0") + self.idx("v_0")
        basic_ode = self.get_basic_ode()

        def odefun(t, y):
            base_result = basic_ode(t, y)
            if self.ps["t_start"] <= t[0] < self.ps["t_start"] + self.ps["T"]:
                v_div[:, div_idx] = (y @ self.V_2)[:, div_idx]
                vacc = torch.div(V_1_mul(y, self.V_1), v_div)
                return base_result + vacc
            return base_result

        return odefun

    def _get_V_1_from_lhs(self, lhs_table):
        daily_vacc = (lhs_table * self.ps["total_vaccines"] / self.ps["T"]).to(self.device)
        lhs_dict = {"daily_vac": daily_vacc}
        return self.get_matrix_from_lhs(lhs_dict=lhs_dict, matrix_name="V_1")
