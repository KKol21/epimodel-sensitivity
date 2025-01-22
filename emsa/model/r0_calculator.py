import torch

from .matrix_generator import (
    generate_transition_matrix,
    get_susc_mul,
    get_inf_mul,
    get_param_mul,
)
from . import get_substates
from typing import Dict, Any


class R0Generator:
    def __init__(self, data, model_struct: Dict[str, Any]):
        """
        This class generates the basic reproduction number (R0) for the epidemic model.

        Args:
            data (Any): Data for the epidemic model.
            model_struct (Dict[str, Any]): Structure of the epidemic model.
        """
        self.data = data
        self.device = data.device
        self.state_data = model_struct["state_data"]
        self.trans_data = model_struct["trans_data"]
        self.tms_rules = model_struct["tms_rules"]

        self.inf_states = self.get_infected_states()
        self.n_comp = len(self.inf_states)
        self.n_age = len(data.age_data.flatten())
        self.n_states = len(self.inf_states)
        self.params = data.params
        self.i = {self.inf_states[index]: index for index in torch.arange(0, self.n_states)}
        self.s_mtx = self.n_age * self.n_states

        self._get_e()

    def isinf_state(self, state):
        return state in [
            state for state, data in self.state_data.items() if data.get("type") == "infected"
        ]

    def get_infected_states(self):
        """
        Get the list of infected states.

        Returns:
            list: List of infected states.
        """
        states = []
        for state, data in self.state_data.items():
            if self.isinf_state(state):
                states += get_substates(data.get("n_substates", 1), state)
        return states

    def _idx(self, state: str) -> torch.BoolTensor:
        """
        Get a BoolTensor representing the indices of the compartments corresponding to a given state.

        Args:
            state (str): State name.

        Returns:
            torch.BoolTensor: Index tensor.
        """
        return torch.arange(self.n_age * self.n_states) % self.n_states == self.i[state]

    def get_eig_val(
        self,
        susceptibles: torch.Tensor,
        population: torch.Tensor,
        contact_mtx: torch.Tensor,
    ) -> float:
        """
        Compute the dominant eigenvalue of the next-generation matrix (NGM).

        Args:
            susceptibles (torch.Tensor): Susceptible population.
            population (torch.Tensor): Total population.
            contact_mtx (torch.Tensor): Contact matrix.

        Returns:
            float: Dominant eigenvalue representing the basic reproduction number (R0).
        """
        # contact matrix needed for effective reproduction number: [c_{j,i} * S_i(t) / N_i(t)]
        cm = contact_mtx / population.reshape((-1, 1))
        cm = cm * susceptibles
        f = self._get_f(cm)
        v_inv = self._get_v()
        ngm_large = v_inv @ f
        ngm = self.e @ ngm_large @ self.e.T if self.n_age > 1 else self.e @ ngm_large @ self.e

        if self.n_age == 1:
            dom_eig_val = torch.abs(ngm)
        else:
            dom_eig_val = torch.sort(torch.abs(torch.linalg.eigvals(ngm)))[0][-1]
        return float(dom_eig_val)

    def _get_v(self) -> torch.Tensor:
        """
        Compute the inverse of the transition matrix.

        Returns:
            torch.Tensor: Inverse of the transition matrix.
        """
        isinf = self.isinf_state
        inf_state_dict = {
            state: data for state, data in self.state_data.items() if isinf(state=state)
        }
        trans_mtx = generate_transition_matrix(
            states_dict=inf_state_dict,
            parameters=self.params,
            n_age=self.n_age,
            n_comp=self.n_states,
            c_idx=self.i,
        ).to(self.device)

        end_state_dict = {
            state: f"{state}_{data.get('n_substates', 1) - 1}"
            for state, data in self.state_data.items()
        }
        inf_trans = [
            trans
            for trans in self.trans_data
            if isinf(state=trans["source"]) and isinf(state=trans["target"])
        ]
        idx = self._idx

        for trans in inf_trans:
            source = trans["source"]
            target = f"{trans['target']}_0"
            param = self.params[self.state_data[source]["rate"]]
            n_substates = self.state_data[source].get("n_substates", 1)
            distr = get_param_mul(trans_params=trans.get("params"), params=self.params)
            trans_mtx[idx(end_state_dict[source]), idx(target)] = param * distr * n_substates
        return torch.linalg.inv(trans_mtx)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        """
        Compute the matrix representing the rate of infection.

        Args:
            contact_mtx (torch.Tensor): The contact matrix.

        Returns:
            torch.Tensor: The matrix representing the rate of infection.
        """
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states
        f = torch.zeros((s_mtx, s_mtx)).to(self.device)

        for tms in self.tms_rules:
            susc_mul = get_susc_mul(tms_rule=tms, data=self.data)
            inf_mul = get_inf_mul(tms_rule=tms, data=self.data)
            for actor in tms["actors-params"]:
                rel_inf = self.params.get(tms["actors-params"][actor], 1)
                for substate in get_substates(
                    n_substates=self.state_data[actor].get("n_substates", 1),
                    comp_name=actor,
                ):
                    substate_slice = slice(i[substate], s_mtx, n_states)
                    target_slice = slice(i[f"{tms['target']}_0"], s_mtx, n_states)
                    f[substate_slice, target_slice] = (
                        susc_mul * contact_mtx.T * inf_mul.unsqueeze(0) * rel_inf
                    )
        return f

    def _get_e(self):
        """
        Compute and store the matrix 'e' used in the next-generation matrix (NGM) calculation.
        """
        block = torch.zeros(
            self.n_states,
        ).to(self.device)
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
