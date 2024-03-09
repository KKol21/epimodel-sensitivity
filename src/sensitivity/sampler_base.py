import os
import time
from abc import ABC, abstractmethod

import numpy as np
from smt.sampling_methods import LHS

from src.sensitivity.target_calc.output_generator import OutputGenerator


class SamplerBase(ABC):
    """
    Base class for performing sampling-based simulations.

    This abstract class defines the common structure and interface for samplers used to explore
    parameter combinations and collect simulation results.

    Args:
        sim_state (dict): The state of the simulation as a dictionary containing relevant params.
        sim_obj: The simulation object representing the underlying simulation model.

    Attributes:
        sim_obj: The simulation object representing the underlying simulation model.
        sim_state (dict): The state of the simulation as a dictionary containing relevant params.
        lhs_boundaries (dict): The boundaries for Latin Hypercube Sampling (LHS) parameter ranges.

    Methods:
        run_sampling(): Runs the sampling-based simulation to explore different parameter combinations
            and collect simulation results.
    """

    def __init__(self, sim_obj, sim_option):
        self.sim_obj = sim_obj
        self.sim_option = sim_option
        self._process_sampling_config()

    def _process_sampling_config(self):
        sim_obj = self.sim_obj
        self.n_samples = sim_obj.n_samples
        self.batch_size = sim_obj.batch_size

        params_bounds = sim_obj.sampled_params_boundaries
        if all([param in sim_obj.params for param in params_bounds.keys()]):
            self.lhs_boundaries = params_bounds

    @abstractmethod
    def run_sampling(self):
        pass

    def _get_lhs_table(self):
        bounds = np.array([bounds for bounds in self.lhs_boundaries.values()])
        sampling = LHS(xlimits=bounds)
        return sampling(self.n_samples)

    def _get_sim_output(self, lhs_table):
        print(f"\n Simulation for {self.n_samples} samples ({self.sim_obj.get_filename(self.sim_option)})")
        print(f"Batch size: {self.batch_size}\n")

        output_generator = OutputGenerator(self.sim_obj, self.sim_option)
        sim_outputs = output_generator.get_output(lhs_table=lhs_table)

        time.sleep(0.3)

        # Save samples, target values
        filename = self.sim_obj.get_filename(self.sim_option)
        self.save_output(output=lhs_table, output_name='lhs', filename=filename)
        for target_var, sim_output in sim_outputs.items():
            self.save_output(output=sim_output.cpu(), output_name='simulations', filename=filename + f"_{target_var}")

    def save_output(self, output, output_name, filename):
        # Create directories for saving calculation outputs
        folder_name = self.sim_obj.folder_name
        os.makedirs(folder_name, exist_ok=True)

        # Save LHS output
        os.makedirs(f"{folder_name}/{output_name}", exist_ok=True)
        filename = f"{folder_name}/{output_name}/{output_name}_{filename}"
        np.savetxt(fname=filename + ".csv", X=output, delimiter=";")


def create_latin_table(n_of_samples, lower, upper):
    bounds = np.array([lower, upper]).T
    sampling = LHS(xlimits=bounds)
    return sampling(n_of_samples)
