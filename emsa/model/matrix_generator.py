from typing import List, Any, Dict, Callable

import torch

from . import get_substates, EpidemicModelBase


def generate_transition_block(transition_param: float, n_states: int) -> torch.Tensor:
    """
    Generate a transition block for the transition matrix.

    Args:
        transition_param: The transition parameter value.
        n_states: The number of states in the block.

    Returns:
        torch.Tensor: The transition block.

    """
    trans_block = torch.zeros((n_states, n_states))
    # Outflow from states (diagonal elements)
    trans_block = trans_block.fill_diagonal_(-transition_param * n_states)
    # Inflow to states (elements under the diagonal)
    trans_block[: n_states - 1, 1:] = trans_block[: n_states - 1, 1:].fill_diagonal_(
        transition_param * n_states
    )
    return trans_block


def generate_transition_matrix(
    states_dict: Dict[str, Any],
    parameters: Dict[str, float],
    n_age: int,
    n_comp: int,
    c_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Generate the transition matrix for the model.

    Args:
        states_dict (Dict[str, Any]): Dictionary of states.
        parameters (Dict[str, float]): A dictionary containing model parameters.
        n_age (int): The number of age groups.
        n_comp (int): The number of compartments.
        c_idx (Dict[str, int]): A dictionary containing the indices of different compartments.

    Returns:
        torch.Tensor: The transition matrix.
    """
    trans_matrix = torch.zeros((n_age * n_comp, n_age * n_comp))
    for age_group in range(n_age):
        for state, data in states_dict.items():
            n_states = data.get("n_substates", 1)
            trans_param = parameters[data["rate"]]
            diag_idx = age_group * n_comp + c_idx[f"{state}_0"]
            block_slice = slice(diag_idx, diag_idx + n_states)
            # Fill in transition block of each transitional state
            trans_matrix[block_slice, block_slice] = generate_transition_block(
                trans_param, n_states
            )
    return trans_matrix


def get_susc_mul(tms_rule: dict, data) -> torch.Tensor:
    """
    Get the susceptibility multiplier.

    Args:
        tms_rule (Dict[str, Any]): Transmission rule.
        data (Any): Model data.

    Returns:
        torch.Tensor: Susceptibility multiplier.
    """
    susc_mul = torch.ones(data.n_age).to(data.device)
    for susc_param in tms_rule.get("susc_params", []):
        susc_mul *= data.params[susc_param]
    return susc_mul


def get_inf_mul(tms_rule: dict, data) -> torch.Tensor:
    """
    Get the infection multiplier.

    Args:
        tms_rule (Dict[str, Any]): Transmission rule.
        data (Any): Model data.

    Returns:
        torch.Tensor: Infection multiplier.
    """
    inf_mul = torch.ones(data.n_age).to(data.device)
    for inf_param in tms_rule.get("infection_params", []):
        inf_mul *= data.params[inf_param]
    return inf_mul


def get_param_mul(trans_params: List[str], params: dict) -> float:
    """
    Get the transition parameters multiplier.

    Args:
        trans_params (List[str]): Distribution parameters keys.
        params (Dict[str, float]): Model parameters.

    Returns:
        float: Transition parameters  multiplier.
    """
    if not trans_params:
        return 1
    result = 1
    for trans_param in trans_params:
        if trans_param[-1] == "_":
            result *= 1 - params[trans_param[:-1]]
        else:
            result *= params[trans_param]
    return result


class MatrixGenerator:
    """
    Class responsible for generating the matrices used in the model.

    Let y be size n_samples * n_eq, each row corresponding to a different simulation. The general formula we represent
    the system of ODEs with is the following:

            y' = (y @ T_1) * (y @ T_2) + y @ L,

    where T_1 and T_2 are responsible for the transmission of the disease, and L defines the linear changes.

    In addition to this, continuous vaccination can be written as

            vacc = (y @ V_1) / (y @ V_2).

    Attributes:
        cm (torch.Tensor): The contact matrix.
        ps (Dict[str, float]): A dictionary containing model parameters.
        data (Any): Model data.
        state_data (Dict[str, Any]): Data related to states.
        trans_data (List[Dict[str, Any]]): Data related to transitions.
        tms_rules (List[Dict[str, Any]]): Transmission rules.
        n_eq (int): The total number of compartments in the model.
        n_age (int): The number of age groups.
        n_comp (int): The number of states.
        population (torch.Tensor): The total population.
        device (torch.device): The device to be used for computations.
        idx (Dict[str, int]): A dictionary containing the indices of different compartments.
        c_idx (Dict[str, int]): A dictionary containing the indices of different compartments' components.

    Methods:
        _get_comp_slice(comp): Get a slice representing the indices of a given compartment.
        _get_end_state(comp): Get the string representing the last state of a given compartment.
        _get_trans_param_dict(): Get a dictionary of transition params for different compartments.
    """

    def __init__(self, model: EpidemicModelBase, cm):
        """
        Initialize the MatrixGenerator instance.

        Args:
            model (EpidemicModelBase): An instance of the EpidemicModelBase class.
            cm: The contact matrix.

        """
        self.cm = cm
        self.ps = model.ps
        self.data = model.data
        self.state_data = model.state_data
        self.trans_data = model.trans_data
        self.tms_rules = model.tms_rules
        self.n_eq = model.n_eq
        self.n_age = model.n_age
        self.n_comp = model.n_comp
        self.population = model.population
        self.device = model.device
        self.idx = model.idx
        self.c_idx = model.c_idx

    def generate_matrix(self, matrix_name: str) -> torch.Tensor:
        matrix_methods = {
            "A": self.get_A,
            "T": self.get_T,
            "B": self.get_B,
            "V_1": self.get_V_1,
            "V_2": self.get_V_2,
        }
        if matrix_name in matrix_methods:
            get_matrix: Callable[[], torch.Tensor] = matrix_methods[matrix_name]
            return get_matrix()
        else:
            raise Exception("Not a valid matrix!")

    def get_A(self) -> torch.Tensor:
        """
        Returns:
            Torch.Tensor: When multiplied with y, the resulting tensor contains the rate of transmission for
            the susceptibles of age group i at the indices of compartments s^i and e_0^i
        """
        A = torch.zeros((self.n_eq, self.n_eq)).to(self.device)
        idx = self.idx

        for tms in self.tms_rules:
            source = f"{tms['source']}_0"
            target = f"{tms['target']}_0"
            susc_mul = get_susc_mul(tms_rule=tms, data=self.data)
            transmission_rate = susc_mul / self.population
            A[idx(source), idx(source)] = -transmission_rate
            A[idx(source), idx(target)] = transmission_rate
        return A

    def get_T(self, cm=None) -> torch.Tensor:
        T = torch.zeros((self.n_eq, self.n_eq)).to(self.device)
        if cm is None:
            cm = self.cm
        for tms in self.tms_rules:
            source = f"{tms['source']}_0"
            target = f"{tms['target']}_0"
            inf_mul = get_inf_mul(tms_rule=tms, data=self.data)
            infection_spread_rate = (
                self.ps["beta"] * torch.atleast_2d(cm).T * inf_mul.unsqueeze(0)
            )  # Broadcast inf_mul to columns
            for actor in tms["actors-params"].keys():
                rel_inf = self.ps.get(tms["actors-params"][actor], 1)
                for substate in get_substates(
                    n_substates=self.state_data[actor].get("n_substates", 1),
                    comp_name=actor,
                ):
                    T[self._get_comp_slice(substate), self._get_comp_slice(source)] = (
                        infection_spread_rate * rel_inf
                    )
                    T[self._get_comp_slice(substate), self._get_comp_slice(target)] = (
                        infection_spread_rate * rel_inf
                    )
        return T

    def get_B(self) -> torch.Tensor:
        ps = self.ps
        state_data = self.state_data
        trans_data = self.trans_data

        # We begin by filling in the transition blocks of the intermediate states
        intermediate_states = {
            state: data
            for state, data in state_data.items()
            if data.get("type", "")
            not in [
                "susceptible",
                "recovered",
                "dead",
            ]
        }
        B = generate_transition_matrix(
            states_dict=intermediate_states,
            parameters=ps,
            n_age=self.n_age,
            n_comp=self.n_comp,
            c_idx=self.c_idx,
        ).to(self.device)

        # Fill in the rest of the first-order terms
        idx = self.idx
        end_state = {
            state: f"{state}_{data.get('n_substates', 1) - 1}" for state, data in state_data.items()
        }
        # Iterate over the linear transitions
        for trans in [trans for trans in trans_data if trans.get("type", "basic") == "basic"]:
            source = trans["source"]
            trans_param = ps[state_data[source]["rate"]]
            # Multiply the transition parameter by the parameter(s) given
            trans_param = trans_param * get_param_mul(
                trans.get("params"), self.ps
            )  # Throws shape error with *= in some cases
            target = f"{trans['target']}_0"
            n_substates = state_data[trans["source"]].get("n_substates", 1)
            B[idx(end_state[source]), idx(target)] = trans_param * n_substates
        return B

    def get_V_1(self, daily_vac=None) -> torch.Tensor:
        if daily_vac is None:
            daily_vac = self.ps["daily_vac"]
        V_1 = torch.zeros((self.n_eq, self.n_eq)).to(self.device)
        # Tensor responsible for the nominators of the vaccination formula
        V_1[self.idx("s_0"), self.idx("s_0")] = daily_vac
        V_1[self.idx("s_0"), self.idx("v_0")] = daily_vac
        return V_1

    def get_V_2(self) -> torch.Tensor:
        idx = self.idx
        V_2 = torch.zeros((self.n_eq, self.n_eq)).to(self.device)
        # Fill in all the terms such that we will divide the terms at the indices of s^i and v^i by (s^i + r^i)
        V_2[idx("s_0"), idx("s_0")] = -1
        V_2[idx("r_0"), idx("s_0")] = -1
        V_2[idx("s_0"), idx("v_0")] = 1
        V_2[idx("r_0"), idx("v_0")] = 1
        return V_2

    def _get_comp_slice(self, comp: str) -> slice:
        """
        Get a slice representing the indices of a given compartment.

        Args:
            comp (str): The compartment name.

        Returns:
            slice: A slice representing the indices of the compartment.
        """
        return slice(self.c_idx[comp], self.n_eq, self.n_comp)

    def get_end_state(self, comp: str) -> str:
        """
        Get the string representing the last state of a given compartment.

        Args:
            comp (str): The compartment name.

        Returns:
            str: The last state of the compartment.
        """
        return f"{comp}_{self.state_data[comp].get('n_substates', 1) - 1}"
