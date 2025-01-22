Format of the inputs
####################


Model Structure
***************

To define the compartments and transitions in your epidemic model, you can refer to the
:doc:`Model structure templates <./model_struct>`. These templates will guide you through how to create
states, transitions, and transmission rules for your model configuration.


Sampling configuration
**********************

Other than the previously described inputs, we will also need to provide a sampling configuration for EMSA.

- **target_vars:** List of target variables to be analyzed during sensitivity analysis.

    - Example: ``["i_max", "d_sup", "r0"]``.
    - `comp_max`: The maximum value of a model compartment, aggregated by age.
    - `comp_sup`: The supremum value of a compartment, aggregated by age.
    - `r0`: The base reproduction number.

- **batch_size:** Number of samples to evaluate in a single batch.
- **n_samples:** Total number of samples used for sensitivity analysis.
- **init_vals:** Dictionary of initial values

    - The dictionary should contain keys representing compartment names and values
      representing the initial population of each compartment.
    - For the susceptible compartment, the values are derived by
      subtracting these initial values from the total age vector.

- (Optional) **is_static:** Whether the total population of the model is static, eg. there are no birth/death mechanisms, aging, etc. If not set to false, but the population size changes, a warning shall be triggered.
- (Optional) **sampled_parameters_boundaries:**


Model data
**********


Model Parameters
================

Model parameters represent the rates or probabilities that govern transitions between states in the epidemic model.
They are provided as key-value pairs where the key represents the parameter name and the value is the corresponding
numeric value (e.g., a rate of recovery, transmission probability, etc.).

Example of model parameters in JSON format:

.. code-block:: json

   {
     "gamma": 0.2,   // Recovery rate
     "alpha": 0.3    // Incubation rate
   }

- **gamma**: This parameter could represent the recovery rate for an infected individual transitioning to the recovered state.
- **alpha**: This parameter might represent the rate at which individuals leave the incubation stage and become infectious.


Age Vector
==========

This is a 1D array (or list) representing the population distribution across different age groups.
Each entry in the vector corresponds to the proportion or number of individuals in a specific age group.


Contact matrix
==============

The contact matrix is a 2D array representing interaction rates between different age groups. Symmetrization is
performed to ensure that the interaction rates between age groups are mathematically correct, since the data
is collected through surveys, inconsistent values are nearly always introduced.

Symmetrization Process:
-----------------------

1. **Averaging with Transpose**: The contact matrix is symmetrized by averaging the element at position `C[i, j]`
with the corresponding element at `C[j, i]`. This step ensures that the interaction rate between age group `i` and
age group `j` is consistent from both perspectives.

2. **Adjustment by Population Size**: After averaging, each row of the matrix is normalized by dividing it by
the size of the age group corresponding to that row. This step accounts for differences in the population sizes
of each age group.

After these operations, the following will hold:

   .. math::

     C'[i, j] = \frac{C[i, j] N_i + C[i, j] N_j}{2N_i},

where the element `C[i, j]` represents the *average number of interactions* a member of age
group `i` has with members of age group `j`. Likewise, `C[j, i]` represents the average number of interactions
a member of age group `j` has with members of age group `i`.
