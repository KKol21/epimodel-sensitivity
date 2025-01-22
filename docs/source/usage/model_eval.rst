Model evaluation
################

Assuming all the inputs necessary, as discussed in the previous section
(:doc:`General use <./general_use>`) are ready and available, the following
will guide you in evaluating your model and creating a figure from the result.


Evaluating the results
======================


    .. code-block:: python


        import matplotlib.pyplot as plt
        import torch

        from emsa.model.epidemic_model import EpidemicModel


        data = ...
        model_struct = ...

        model = EpidemicModel(data=data, model_struct=model_struct)

        # Evaluation timestamps
        t_eval = torch.linspace(1, 300, 300).to(data.device)
        # Initial values
        iv_dict = {"e": [10]}
        iv = model.get_initial_values_from_dict(iv_dict)
        susceptibles = iv[model.idx("s_0")]
        # Evaluate model
        sol = model.get_solution(y0=iv, t_eval=t_eval).ys[0, :, :]

From this we can use the methods implemented by the EpidemicModelBase class to extract values of
specific compartments, and create a figure of our simulation


Plotting the results
====================


    .. code-block:: python


        # Iterate over compartments and plot them aggregated by age
        # (Erlang distributed states are aggregated as well)
        for comp in model.state_data.keys():
            # Aggregate compartments by age:
            # returns compartments values in case of a single age group
            comp_sol = model.aggregate_by_age(sol, comp)
            # Plot compartments values and annotate them
            ax.plot(t, comp_sol, label=comp.upper(), linewidth=2)
        # Pyplot magic
        ax.set_xlabel("Days")
        ax.set_ylabel("Population")
        plt.legend()
        plt.show()
