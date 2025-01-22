import torch
import torchode as to

from .model_base import EpidemicModelBase


class EpidemicModel(EpidemicModelBase):
    def __init__(self, data, model_struct: dict):
        """
        Initialize the EpidemicModel, a class for running single simulations of a given model.

        Parameters:
            data: Model data.
            model_struct (dict): The structure of the model.
        """
        super().__init__(data, model_struct)

    def get_solution(self, y0: torch.Tensor, t_eval: torch.Tensor, **kwargs) -> to.Solution:
        """
        Get the solution of the ODE using the initial conditions and evaluation times,
        using the ODE solver in EpidemicModelBase .

        Parameters:
            y0 (torch.Tensor): Initial state.
            t_eval (torch.Tensor): Times at which to evaluate the solution.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Solution of the ODE.
        """
        self.initialize_matrices()
        odefun = kwargs.get("odefun", self.basic_ode)
        return self.get_sol_from_ode(
            y0=torch.atleast_2d(y0),
            t_eval=torch.atleast_2d(t_eval),
            odefun=odefun,
        )

    def basic_ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Basic ODE function without vaccination.

        Parameters:
            t (torch.Tensor): Current time.
            y (torch.Tensor): Current state.

        Returns:
            torch.Tensor: Derivative of the system.
        """
        return torch.mul(y @ self.A, y @ self.T) + y @ self.B
