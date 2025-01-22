import json
from os.path import join

import torch
import xlrd

from emsa.utils.dataloader import DataLoaderBase


class DataLoader(DataLoaderBase):
    def __init__(self, params_path=None, contact_data_path=None, age_data_path=None):
        super().__init__()
        self._model_parameters_data_file = (
            join(self.project_path, "emsa_examples/data/model_parameters.json")
            if params_path is None
            else params_path
        )
        self._contact_data_file = (
            join(self.project_path, "emsa_examples/data/contact_matrices.xls")
            if contact_data_path is None
            else contact_data_path
        )
        self._age_data_file = (
            join(self.project_path, "emsa_examples/data/age_distribution.xls")
            if age_data_path is None
            else age_data_path
        )

        self._get_age_data()
        self._get_model_parameters_data()
        self._get_contact_mtx()

        self.device = "cpu"

    def _get_age_data(self):
        wb = xlrd.open_workbook(self._age_data_file)
        sheet = wb.sheet_by_index(0)
        datalist = torch.Tensor([sheet.row_values(i) for i in range(0, sheet.nrows)])
        wb.unload_sheet(0)
        self.age_data = datalist.to(self.device)

    def _get_model_parameters_data(self):
        # Load model param_names
        with open(self._model_parameters_data_file) as f:
            parameters = json.load(f)
        self.params = dict()
        for param in parameters.keys():
            param_value = parameters[param]["value"]
            if isinstance(param_value, list):
                self.params.update({param: torch.Tensor(param_value).to(self.device)})
            else:
                self.params.update({param: param_value})

    def _get_contact_mtx(self):
        wb = xlrd.open_workbook(self._contact_data_file)
        contact_matrices = dict()
        for idx in range(4):
            sheet = wb.sheet_by_index(idx)
            datalist = torch.Tensor([sheet.row_values(i) for i in range(0, sheet.nrows)]).to(
                self.device
            )
            cm_type = wb.sheet_names()[idx]
            wb.unload_sheet(0)
            datalist = self.transform_matrix(datalist)
            contact_matrices.update({cm_type: datalist})
        self.cm = (
            contact_matrices["home"]
            + contact_matrices["work"]
            + contact_matrices["school"]
            + contact_matrices["other"]
        )

    def transform_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Perform symmetrization of contact matrix, by averaging its elements with the elements of
        it's transpose, then dividing each row with the size of the age group corresponding to the given row.

        After this, the C[i, j] element represents the average number of interactions a member of age
        group i has with members of age group j, while C[j, i] represents the average number of
        interactions a member of age group j has with members of age group j in a day.

        For example, in the case that the population of age group i is the double of age group j:

                                     C[j, i] = 2 * C[i, j]

        Args:
            matrix (torch.Tensor): The contact matrix.

        Returns:
            torch.Tensor: The transformed contact matrix.
        """
        # Age vector as a column vector
        age_distribution = self.age_data.reshape((-1, 1))
        # Matrix containing total number of contacts
        total_contact = matrix * age_distribution
        # Symmetrize matrix
        output = (total_contact + total_contact.T) / 2
        # Divide by age group sizes
        output /= age_distribution
        return output
