Extending the basic model format
################################


The model representation used as the basis for the solver implemented in EMSA is limited to basic compartmental models.
Because of this, EMSA can handle by default:

1. Linear transitions
2. Erlang transitions
3. Mass-effect based transmission mechanisms

In this section, we will take a look at how a model beyond the basic format can be implemented.

Continuous vaccination
======================

In order to simulate more sophisticated models, you will need to extend the model format yourself. We will discuss
the model in the vaccinated_sensitivity example, which is an implementation of the following ODE system, where
i denotes the i-th age group:

.. math::

    \begin{align*}
        {S^i}' &= - \text{transmission} - v (t) \cdot \theta_i \cdot \frac{S^i}{S^i + R^i} \\
        \vdots\\
        {V^i}' &= v (t) \cdot \theta_i \cdot \frac{S^i}{S^i + R^i}
    \end{align*}

where :math:`v(t)` is 1 inside the vaccination period, otherwise 0, :math:`\theta_i` is the daily vaccines given to age group
i.

Using the methods defined in MatrixGenerator, we can create the necessary matrices for representing these equations:

.. code-block:: python

    def get_vaccinated_ode(self, curr_batch_size):
        V_1_mul = self.get_mul_method(self.V_1)

        v_div = torch.ones((curr_batch_size, self.n_eq)).to(self.device)
        div_idx = self.idx("s_0") + self.idx("v_0")
        basic_ode = self.get_basic_ode()

        def odefun(t, y):
            base_result = basic_ode(t, y)
            if self.ps["t_start"] <= t[0] < self.ps["t_start"] + self.ps["T"]:
                v_div[:, div_idx] = (y @ self.V_2)[:, div_idx]
                vacc = torch.div(V_1_mul(y, self.V_1), v_div)
                return base_result + vacc
        return base_result
    return odefun


If using matrices to represent our desired model is infeasible or too difficult, we can also manually extract the
required compartments values. In our example, that would look like this:
