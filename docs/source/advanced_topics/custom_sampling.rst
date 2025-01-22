Sampling form non-uniform distributions
#######################################

If we want to represent regions of our parameter space to different degrees, we can use non-uniform distributions
for sampling. In such cases, we can build our sampler by inheriting from SamplerBase and implementing
our custom sampling logic there.

Note: this can also introduce bias to the sensitivity of the parameters, thus it should be handled with care.



Example of a custom sampling scenario
=====================================

This example demonstrates how to sample three parameters: `alpha`, `beta`, and `gamma`. The `alpha` and `beta`
parameters are uniformly distributed, while `gamma` follows a standard normal distribution, which is scaled to
fit between 0 and 5.


Asssumptions
------------

1. The sampled parameters are provided in sampling_config if the model evaluation can be handled by EMSA (i.e. we sample "basic" parameters),

or

2. The SamplerBase.lhs_bounds member variable contains the sampling boundaries in the correct format. (for an example of the 2. option, see ContactSampler or VaccinatedSampler in emsa_examples)

We also assume that gamma was sampled in between 0 and 1. Then the following steps will provide us the desired result:


Steps
-----

1. Use `SamplerBase.get_lhs_table()` to generate a Latin Hypercube Sampling (LHS) table with uniformly distributed values for all parameters.
2. Create a function that converts uniformly distributed values on [0, 1] into normally distributed values on the interval [0, 5].
3. Apply this function to the `gamma` values in the LHS table.


Implementation
--------------

.. code-block:: python


    from scipy.stats import norm
    from emsa.sensitivity import SamplerBase
    import numpy as np


    def uniform_to_normal_on_interval(data, std=1):
        # Convert uniform [0, 1] to standard normal using the inverse CDF (Probit function)
        normal_values = norm.ppf(data)

        # Apply the logistic transformation to map to (0, 5)
        transformed_values = 5 / (1 + np.exp(-std * normal_values))

        return transformed_values


    class CustomSampler(SamplerBase):
        def __init__(self, sim_object, variable_params=None):
            super().__init__(sim_object, variable_params)

        def run(self):
            lhs_table = self.get_lhs_table()
            # Note: if gamma was age dependent, we can use a slice object to perform the following
            gamma_idx = 2
            # Apply the sampling transformation
            transformed_gamma = uniform_to_normal_on_interval(lhs_table[:, gamma_idx])
            # Overwrite old values with the transformed ones
            lhs_table[:, gamma_idx] = transformed_gamma
            # Calculate targets, etc.
            self.get_sim_output(lhs_table)
