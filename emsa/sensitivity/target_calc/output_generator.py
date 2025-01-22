import torch
import numpy as np

from .r0_calculator_lhs import R0CalculatorLHS
from .sol_based_target_calc import TargetCalc
from typing import Dict

from emsa.utils.simulation_base import SimulationBase


class OutputGenerator:
    def __init__(self, sim_object: SimulationBase):
        self.batch_size = sim_object.batch_size
        self.sim_object = sim_object

    def get_output(self, lhs_table: np.ndarray) -> Dict[str, torch.Tensor]:
        lhs = torch.from_numpy(lhs_table).float().to(self.sim_object.device)
        targets = self.sim_object.target_vars
        output = {}
        target_endings = [target[-3:] for target in targets if target != "r0"]

        if "max" in target_endings or "sup" in target_endings:
            target_calc = TargetCalc(
                model=self.sim_object.model,
                targets=targets,
                config=self.sim_object.target_calc_config,
            )
            sol_based_output = target_calc.get_output(lhs_table=lhs, batch_size=self.batch_size)
            output.update(sol_based_output)

        if "r0" in targets:
            r0calc = R0CalculatorLHS(self.sim_object)
            output["r0"] = r0calc.get_output(lhs_table=lhs)
        return output
