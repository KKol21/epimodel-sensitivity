import torch
from tqdm import tqdm

from emsa.model import R0Generator
from emsa.sensitivity import get_params_col_idx, get_lhs_dict
from emsa.utils import SimulationBase


class R0CalculatorLHS:
    """
    R0Calculator class for calculating the basic reproduction number (R0) for epidemic models.

    Attributes:
        sim_object: An instance of SimulationBase containing simulation parameters and data.
    """

    def __init__(self, sim_object: SimulationBase):
        self.sim_object = sim_object

    def get_output(self, lhs_table: torch.Tensor) -> torch.Tensor:
        """
        Calculate the R0 values for the given LHS (Latin Hypercube Sampling) table.

        Parameters:
            lhs_table (torch.Tensor): LHS table with sampled parameter values.

        Returns:
            torch.Tensor: Calculated R0 values for each sample.

        Raises:
            ValueError: If sampled parameters boundaries are not specified in the simulation object.
        """
        sim_object = self.sim_object
        n_samples = lhs_table.shape[0]
        spb = sim_object.sampled_params_boundaries
        if spb is None:
            raise ValueError(
                "Sampled parameters boundaries not specified, automatic R0 generation isn't possible"
            )
        pci = get_params_col_idx(sampled_params_boundaries=spb)
        lhs_dict = get_lhs_dict(params=spb.keys(), lhs_table=lhs_table, params_col_idx=pci)
        r0gen = R0Generator(sim_object.data, sim_object.model_struct)
        r0s = []
        print(f"Calculating R0 for {n_samples} samples")
        for idx in tqdm(range(n_samples)):
            # Select idx. value from lhs table for each parameter
            row_dict = {
                key: value[idx] if len(value.size()) < 2 else value[idx, :]
                for key, value in lhs_dict.items()
            }
            r0gen.params.update(row_dict)
            r0 = sim_object.params["beta"] * r0gen.get_eig_val(
                contact_mtx=sim_object.cm,
                susceptibles=sim_object.susceptibles.reshape(1, -1),
                population=sim_object.population,
            )
            r0s.append(r0)
        return torch.tensor(r0s, device=sim_object.device)
