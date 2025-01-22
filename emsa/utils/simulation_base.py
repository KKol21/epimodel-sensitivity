import itertools
import json
import os
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss
from torch import atleast_2d

from .dataloader import PROJECT_PATH
from .plotter import generate_tornado_plot


class SimulationBase(ABC):
    def __init__(self, data, model_struct_path, sampling_config_path):
        # Load data
        self.data = data
        self.device = data.device
        self.model = None

        self._load_simulation_data()
        self._load_config(sampling_config_path)
        self._load_model_structure(model_struct_path)

    @property
    def susceptibles(self):
        return self.model.get_initial_values()[self.model.idx("s_0")]

    def _load_simulation_data(self):
        self.params = self.data.params
        self.cm = atleast_2d(self.data.cm)
        self.population = self.data.age_data.flatten()
        self.n_age = len(self.population)
        self.folder_name = os.path.join(PROJECT_PATH, "sens_data")

    def _load_model_structure(self, model_struct_path):
        with open(model_struct_path) as f:
            self.model_struct = json.load(f)

    def _load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)

        self.target_vars = config["target_vars"]

        self.variable_params_dict = config.get("variable_params", {})

        self.variable_param_combinations = self.process_variable_params()

        self.sampled_params_boundaries = config.get("sampled_params_boundaries") or {}
        self.n_samples = config["n_samples"]
        self.batch_size = config["batch_size"]

        self.test = config.get("is_static") or True
        self.init_vals = config["init_vals"]

        self.target_calc_config = {
            "tlim_ini": config.get("tlim_ini") or 300,
            "tlim_final": config.get("tlim_final") or 5000,
            "tdelta": config.get("tdelta") or 50,
        }

    def process_variable_params(self):
        vpd = self.variable_params_dict

        if len(vpd) == 1:
            key = next(iter(vpd))
            return self.flatten_dict(vpd, key)

        flattened_vpd = [self.flatten_dict(vpd, key) for key in vpd.keys()]
        variable_params_product = list(itertools.product(*flattened_vpd))
        return [self.merge_dicts(variable_params) for variable_params in variable_params_product]

    def flatten_dict(self, d, key):
        if isinstance(d[key], dict):
            return self.flatten_dict_in_dict(d, key)
        else:
            return self.flatten_list_in_dict(d, key)

    @staticmethod
    def flatten_list_in_dict(d, key):
        return [{key: value} for value in d[key]]

    @staticmethod
    def flatten_dict_in_dict(d, key):
        return [{key: {subkey: value}} for subkey, value in d[key].items()]

    @staticmethod
    def merge_dicts(ds):
        d_out = {key: value for d in ds for key, value in d.items()}
        return d_out

    @abstractmethod
    def run_sampling(self):
        pass

    def run_func_for_all_configs(self, func):
        for variable_params, target in itertools.product(
            self.variable_param_combinations, self.target_vars
        ):
            filename = self.get_filename(variable_params) + f"_{target}"
            func(filename)

    def get_filename(self, variable_params):
        return (
            "_".join(
                [self.parse_param_name(variable_params, key) for key in variable_params.keys()]
            )
            if variable_params
            else ""
        )

    @staticmethod
    def parse_param_name(variable_params: dict, key):
        if isinstance(variable_params[key], dict):
            subkey = next(iter(variable_params[key]))
            return f"{key}-{subkey}"
        else:
            return f"{key}-{variable_params[key]}"

    def get_beta_from_r0(self, base_r0):
        from emsa.model import R0Generator

        r0generator = R0Generator(self.data, self.model_struct)
        if isinstance(base_r0, tuple):
            base_r0 = base_r0[0]
        return base_r0 / r0generator.get_eig_val(
            contact_mtx=self.cm,
            susceptibles=self.susceptibles.reshape(1, -1),
            population=self.population,
        )

    def calculate_prcc(self, filename: str, target: str) -> None:
        """

        Calculates PRCC (Partial Rank Correlation Coefficient) values from saved LHS tables and simulation results.

        This method reads the saved LHS tables and simulation results for each parameter combination and calculates
        the PRCC values. The PRCC values are saved in separate files in the 'sens_data_"folder_name"/prcc' directory.

        """
        from emsa.sensitivity import get_prcc_values

        folder_name = self.folder_name
        os.makedirs(os.path.join(folder_name, "prcc"), exist_ok=True)
        lhs_path = os.path.join(folder_name, f"lhs/lhs_{filename}.csv")
        output_path = os.path.join(folder_name, f"simulations/simulations_{filename}_{target}.csv")
        lhs_table = np.loadtxt(lhs_path)
        sim_output = np.loadtxt(output_path)

        prcc = get_prcc_values(np.c_[lhs_table, sim_output.T])

        prcc_path = os.path.join(folder_name, f"prcc/prcc_{filename}_{target}.csv")
        np.savetxt(fname=prcc_path, X=prcc)

    def calculate_p_values(self, filename, significance=0.05):
        p_values_dir = os.path.join(self.folder_name, "p_values")
        os.makedirs(p_values_dir, exist_ok=True)

        prcc_path = os.path.join(self.folder_name, f"prcc/prcc_{filename}.csv")
        prcc = np.loadtxt(fname=prcc_path)
        denom = np.where(prcc**2 < 1 - 1e-6, 1 - prcc**2, 0.01)
        t = prcc * np.sqrt((self.n_samples - 2 - self.n_age) / denom)
        # p-value for 2-sided test
        dof = self.n_samples - 2 - self.n_age
        p_values = 2 * (1 - ss.t.cdf(x=abs(t), df=dof))
        p_values = np.atleast_1d(p_values)

        p_values_path = os.path.join(self.folder_name, f"p_values/p_values_{filename}.csv")
        np.savetxt(fname=p_values_path, X=p_values)

        is_first = True
        if len(p_values) < 30:
            for idx, p_val in enumerate(p_values):
                if p_val > significance:
                    if is_first:
                        print("\nInsignificant p-values in ", filename, " case: \n")
                        is_first = False
                    print(f"\t {idx}. p-val: ", p_val)

    def plot_prcc(self, filename: str, labels=None):
        """

        Args:
            filename:
            labels:

        Returns: None

        """
        spb = self.sampled_params_boundaries
        from emsa.sensitivity.sensitivity_model_base import get_params_col_idx

        def get_aged_param_labels(aged_param):
            return [f"{aged_param}_{ag}" for ag in range(self.n_age)]

        if labels is None:
            labels = []
            pci = get_params_col_idx(sampled_params_boundaries=spb)
            for param, idx in pci.items():
                param_label = (
                    get_aged_param_labels(param) if isinstance(spb[param][0], list) else param
                )
                if isinstance(param_label, list):
                    labels += param_label
                else:
                    labels.append(param_label)

        prcc_plots_dir = os.path.join(self.folder_name, "prcc_plots")
        os.makedirs(prcc_plots_dir, exist_ok=True)

        prcc_file = os.path.join(self.folder_name, f"prcc/prcc_{filename}.csv")
        prcc = np.loadtxt(fname=prcc_file)

        p_values_file = os.path.join(self.folder_name, f"p_values/p_values_{filename}.csv")
        p_val = np.loadtxt(fname=p_values_file)

        generate_tornado_plot(
            sim_object=self, labels=labels, prcc=prcc, p_val=p_val, filename=filename
        )

    def calculate_all_prcc(self):
        for variable_params, target in itertools.product(
            self.variable_param_combinations, self.target_vars
        ):
            filename = self.get_filename(variable_params)
            self.calculate_prcc(filename=filename, target=target)

    def calculate_all_p_values(self):
        self.run_func_for_all_configs(self.calculate_p_values)

    def plot_all_prcc(self):
        self.run_func_for_all_configs(self.plot_prcc)
