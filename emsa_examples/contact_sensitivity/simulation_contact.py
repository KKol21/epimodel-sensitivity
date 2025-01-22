import os

import numpy as np

from emsa_examples.contact_sensitivity.sampler_contact import SamplerContact
from emsa_examples.contact_sensitivity.sensitivity_model_contact import ContactModel
from emsa.utils.plotter import plot_prcc_p_values_as_heatmap
from emsa.utils.simulation_base import SimulationBase, PROJECT_PATH


class SimulationContact(SimulationBase):
    """
    Class for running simulations and analyzing results of sensitivity of the model
    to the contact matrix

    Methods:
        __init__(): Initializes the SimulationVaccinated object.
        run_sampling(): Generates samples, runs simulations, and saves the results.
        calculate_prcc(): Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables.
        plot_prcc(): Creates and saves tornado plots from sensitivity data.
        plot_optimal_vaccine_distributions(): Creates and saves epidemic plots for the most
        optimal vaccine distributions.
        plot_subopt(): Creates and saves epidemic plots for suboptimal vaccine distributions.
        _get_initial_config(): Retrieves initial configurations for the simulation.

    """

    def __init__(self, data):
        config_path = os.path.join(
            PROJECT_PATH,
            "emsa_examples/contact_sensitivity/configs/sampling_config.json",
        )
        model_struct_path = os.path.join(
            PROJECT_PATH, "emsa_examples/contact_sensitivity/configs/model_struct.json"
        )
        super().__init__(
            data=data,
            model_struct_path=model_struct_path,
            sampling_config_path=config_path,
        )

        self.upper_tri_size = int((self.n_age + 1) * self.n_age / 2)
        self.folder_name += "/sens_data_contact"

        self.model = ContactModel(sim_object=self, base_r0=None)

    def run_sampling(self):
        """

        Runs the sampling-based simulation with different parameter combinations.

        This method generates Latin Hypercube Sampling (LHS) samples of vaccine distributions for each parameter
        combination. The LHS tables and simulation results are saved in separate files in the 'sens_data_contact/lhs'
        and 'sens_data_contact/simulations' directories, respectively.

        """
        for variable_params in self.variable_param_combinations:
            # Get beta based on base reproduction number
            base_r0 = variable_params["r0"]
            beta = self.get_beta_from_r0(base_r0=base_r0)
            self.params.update({"beta": beta})

            # Generate matrices used in model representation
            self.model = ContactModel(sim_object=self, base_r0=base_r0)
            self.model.initialize_matrices()

            param_generator = SamplerContact(sim_object=self, variable_params=variable_params)
            param_generator.run()

    def plot_prcc_and_p_values(self, filename):
        """

        Generates and saves PRCC plots based on the calculated PRCC values.

        This method reads the saved PRCC values for each parameter combination and generates
        PRCC plots using the `generate_prcc_plot` function. The plots are saved in separate files
        in the subfolder sens_data_contact/prcc_plots.


        """
        os.makedirs(f"{self.folder_name}/prcc_p_val_plots", exist_ok=True)
        prcc = np.loadtxt(fname=f"{self.folder_name}/prcc/prcc_{filename}.csv")
        p_val = np.loadtxt(fname=f"{self.folder_name}/p_values/p_values_{filename}.csv")

        plot_prcc_p_values_as_heatmap(
            n_age=self.n_age,
            prcc_vector=prcc,
            p_values=p_val,
            filename_to_save=f"{self.folder_name}/prcc_p_val_plots/{filename}.pdf",
            plot_title=f"Contact sensitivity values ({filename})",
        )
