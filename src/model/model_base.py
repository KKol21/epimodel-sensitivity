from abc import ABC, abstractmethod

import torch
import torchode as to


class EpidemicModelBase(ABC):
    def __init__(self, data):
        """
        Initialises Abstract base class for epidemic models.

        This class provides the base functionality for epidemic models. It contains methods to initialize the model,
        retrieve initial values, and obtain the model solutution

        Returns:
            None
        """
        self.data = data
        self.n_age = data.n_age
        self.population = data.age_data.flatten()
        self.compartments = self.get_compartments()
        self.n_comp = len(self.compartments)
        self.ps = data.model_params
        self.device = data.device

        self.c_idx = {comp: idx for idx, comp in enumerate(self.compartments)}
        self.s_mtx = self.n_age * self.n_comp
        self.is_vaccinated = "vaccination" in [trans["type"] for trans in self.data.trans_data.values()]

        from src.model.matrix_generator import MatrixGenerator
        self.matrix_generator = MatrixGenerator(model=self, cm=data.cm)

        self.A = None
        self.T = None
        self.B = None
        self.V_1 = None
        self.V_2 = None

    def initialize_constant_matrices(self):
        mtx_gen = self.matrix_generator
        self.A = mtx_gen.get_A()
        self.T = mtx_gen.get_T()
        self.B = mtx_gen.get_B()
        if self.is_vaccinated:
            self.V_1 = mtx_gen.get_V_1()
            self.V_2 = mtx_gen.get_V_2()

    @abstractmethod
    def get_solution(self, y0, t_eval, **kwargs):
        pass

    def get_sol_from_ode(self, y0, t_eval, odefun):
        term = to.ODETerm(odefun)
        step_method = to.Euler(term=term)
        step_size_controller = to.FixedStepController()
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
        dt0 = torch.full((y0.shape[0],), 1).to(self.device)

        return solver.solve(problem, dt0=dt0)

    def get_compartments(self):
        compartments = []
        for name, data in self.data.state_data.items():
            compartments += get_substates(data["n_substates"], name)
        return compartments

    def get_initial_values(self):
        """

        This method retrieves the initial values for the model. It sets the initial value for the infected compartment
        of the 3rd age group (i_0^3) to 1 and subtracts 1 from the susceptible (s_0^3) compartment for the appropriate
        age group.

        Returns:
            torch.Tensor: Initial values of the model.

        """
        iv = torch.zeros(self.s_mtx).to(self.device)
        age_group = 3
        iv[age_group + self.c_idx['i_0']] = 1
        iv[self.idx('s_0')] = self.population
        iv[age_group * self.n_comp + self.c_idx['s_0']] -= 1
        return iv

    def idx(self, state: str) -> bool:
        return torch.arange(self.s_mtx) % self.n_comp == self.c_idx[state]

    def aggregate_by_age(self, solution, comp):
        """

        This method aggregates the solution by age for a compartment by summing the solution
        values of individual substates for each age group.

        """
        substates = get_substates(n_substates=self.data.state_data[comp]["n_substates"], comp_name=comp)
        substate_indices = [self.idx(state) for state in substates]
        substate_sums = torch.stack([solution[:, idx].sum(dim=1) for idx in substate_indices], dim=0)
        return substate_sums.sum(dim=0)


def get_substates(n_substates, comp_name):
    """
   Args:
       n_substates (int): Number of substates
       comp_name (str): Compartment name.

   Returns:
       list: List of state names.

   """
    return [f"{comp_name}_{i}" for i in range(n_substates)]
