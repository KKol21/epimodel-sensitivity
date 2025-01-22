from abc import ABC

import torch

from emsa.model import EpidemicModelBase


def get_params_col_idx(sampled_params_boundaries: dict):
    params_col_idx = {}
    last_idx = 0
    for param, bound in sampled_params_boundaries.items():
        param_dim = len(bound[0]) if isinstance(bound[0], list) else 1
        params_col_idx[param] = (
            last_idx if param_dim == 1 else slice(last_idx, last_idx + param_dim)
        )
        last_idx += param_dim
    return params_col_idx


def get_lhs_dict(params: list, lhs_table: torch.Tensor, params_col_idx: dict) -> dict:
    return {param: lhs_table[:, params_col_idx[param]] for param in params}


class SensitivityModelBase(EpidemicModelBase, ABC):
    """
    Base class for implementing epidemic models with the capacity
    to run sample based simulations for sensitivity analysis.
    """

    def __init__(self, sim_object):
        super().__init__(data=sim_object.data, model_struct=sim_object.model_struct)
        self.sim_object = sim_object
        self.test = sim_object.test

    def get_basic_ode(self):
        A_mul = self.get_mul_method(self.A)
        T_mul = self.get_mul_method(self.T)
        B_mul = self.get_mul_method(self.B)

        def odefun(t, y):
            return torch.mul(A_mul(y, self.A), T_mul(y, self.T)) + B_mul(y, self.B)

        return odefun

    @staticmethod
    def get_mul_method(tensor: torch.Tensor):
        def mul_by_2d(y, tensor):
            return y @ tensor

        def mul_by_3d(y, tensor):
            return torch.einsum("ij,ijk->ik", y, tensor)

        return mul_by_2d if len(tensor.size()) < 3 else mul_by_3d

    def get_matrix_from_lhs(self, lhs_dict: dict, matrix_name: str):
        n_eq = self.n_eq
        n_samples = next(iter(lhs_dict.values())).shape[0]
        mtx = torch.zeros((n_samples, n_eq, n_eq)).to(self.device)
        ps_original = self.matrix_generator.ps.copy()
        for idx in range(n_samples):
            # Select idx. value from lhs table for each parameter
            row_dict = {
                key: value[idx] if len(value.size()) < 2 else value[idx, :]
                for key, value in lhs_dict.items()
            }
            self.matrix_generator.ps.update(row_dict)

            mtx[idx, :, :] = self.matrix_generator.generate_matrix(matrix_name)
        self.matrix_generator.ps = ps_original
        return mtx

    def get_initial_values(self):
        return self.get_initial_values_from_dict(self.sim_object.init_vals)

    def generate_3D_matrices(self, samples: torch.Tensor):
        spb = self.sim_object.sampled_params_boundaries
        # If the matrix containing the analysed parameter isn't part of the basic representation,
        # the 3D version has to be generated manually. If a matrix isn't used, the desired effect
        # needs to be incorporated somehow else in the odefun used in the solver.
        if spb is None:
            return
        # Params in B
        trans_params = [
            param
            for trans in self.trans_data
            if trans.get("params")
            for param in trans.get("params")
        ]
        trans_rates = [self.state_data[trans["source"]]["rate"] for trans in self.trans_data]
        linear_params = [param for param in spb if param in trans_rates + trans_params]
        # Params in T_1
        susc_params = [
            param for tms_rule in self.tms_rules for param in tms_rule.get("susc_params", [])
        ]
        transmission_params_left = [param for param in spb if param in susc_params]
        # Params in T_2
        actor_params = [
            param for tms_rule in self.tms_rules for param in tms_rule["actors-params"].values()
        ]
        inf_params = actor_params + [
            param for tms_rule in self.tms_rules for param in tms_rule.get("infection_params", [])
        ]
        transmission_params_right = [param for param in spb if param in inf_params + ["beta"]]

        pci = get_params_col_idx(sampled_params_boundaries=spb)

        tpl_lhs = get_lhs_dict(transmission_params_left, samples, pci)
        tpr_lhs = get_lhs_dict(transmission_params_right, samples, pci)
        lp_lhs = get_lhs_dict(linear_params, samples, pci)
        self.A = (
            self.get_matrix_from_lhs(tpl_lhs, "A")
            if len(tpl_lhs) > 0
            else self.matrix_generator.get_A()
        )
        self.T = (
            self.get_matrix_from_lhs(tpr_lhs, "T")
            if len(tpr_lhs) > 0
            else self.matrix_generator.get_T()
        )
        self.B = (
            self.get_matrix_from_lhs(lp_lhs, "B")
            if len(lp_lhs) > 0
            else self.matrix_generator.get_B()
        )
