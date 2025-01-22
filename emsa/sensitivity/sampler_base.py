import os
from abc import ABC, abstractmethod

import numpy as np
from smt.sampling_methods import LHS

from .sensitivity_model_base import get_params_col_idx
from .target_calc import OutputGenerator


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_object: The simulation object representing the underlying simulation model.

    Attributes:
        sim_object: The simulation object representing the underlying simulation model.
        lhs_bounds_dict (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_object, variable_params=None):
        self.sim_object = sim_object
        self.variable_params = variable_params or {}
        self.sampled_params_boundaries = sim_object.sampled_params_boundaries
        self._process_sampling_config()

    def _process_sampling_config(self):
        sim_object = self.sim_object
        self.n_samples = sim_object.n_samples
        self.batch_size = sim_object.batch_size

        if spb := self.sampled_params_boundaries:
            self.lhs_bounds_dict = {param: np.array(spb[param]) for param in spb}
            self.pci = get_params_col_idx(spb)

    @abstractmethod
    def run(self):
        pass

    def get_lhs_table(self):
        bounds = self._get_lhs_bounds()
        sampling = LHS(xlimits=bounds)
        return sampling(self.n_samples)

    def _get_lhs_bounds(self):
        general_bounds = self._get_general_param_bounds()
        age_spec_bounds = self._get_age_spec_param_bounds()

        if age_spec_bounds.shape[0] == 0:
            return general_bounds
        elif general_bounds.shape[0] == 0:
            return age_spec_bounds
        else:
            return self._concat_bounds(
                non_spec_bounds=general_bounds, age_spec_bounds=age_spec_bounds
            )

    def _get_general_param_bounds(self):
        return np.array([bound for bound in self.lhs_bounds_dict.values() if len(bound.shape) == 1])

    def _get_age_spec_param_bounds(self):
        age_spec_bounds = [
            bounds
            for param in self.lhs_bounds_dict
            for bounds in self.lhs_bounds_dict[param]
            if len(self.lhs_bounds_dict[param].shape) == 2
        ]
        return np.array(age_spec_bounds).T

    def _concat_bounds(self, non_spec_bounds: np.ndarray, age_spec_bounds: np.ndarray):
        n_cols = non_spec_bounds.shape[0] + age_spec_bounds.shape[1]
        bounds = np.zeros(shape=(2, n_cols))
        bounds_dict = self.lhs_bounds_dict
        for param, idx in self.pci.items():
            param_bounds = bounds_dict[param]
            bounds[:, idx] = param_bounds
        return bounds.T

    def get_sim_output(self, lhs_table: np.ndarray):
        print(
            f"\n Simulation for {self.n_samples} samples ({self.sim_object.get_filename(self.variable_params)})"
        )
        print(f"Batch size: {self.batch_size}\n")

        output_generator = OutputGenerator(sim_object=self.sim_object)
        sim_outputs = output_generator.get_output(lhs_table=lhs_table)
        if sim_outputs == {}:
            raise Exception("No output was produced by OutputGenerator instance!")

        # Save samples, target values
        filename = self.sim_object.get_filename(self.variable_params)
        self.save_output(output=lhs_table, output_name="lhs", filename=filename)
        for target_var, sim_output in sim_outputs.items():
            self.save_output(
                output=sim_output.cpu(),
                output_name="simulations",
                filename=filename + f"_{target_var}",
            )

    def save_output(self, output, output_name: str, filename: str):
        folder_name = self.sim_object.folder_name
        os.makedirs(folder_name, exist_ok=True)

        dirname = os.path.join(folder_name, output_name)
        filename = os.path.join(dirname, f"{output_name}_{filename}")
        os.makedirs(dirname, exist_ok=True)
        np.savetxt(fname=filename + ".csv", X=output)


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
