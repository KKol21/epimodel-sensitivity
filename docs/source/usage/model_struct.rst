Model structure Templates
=============================

This section explains how to define the different parts of your epidemic model using JSON format.
Below are templates for defining states, transitions, and transmission rules in your model.
You can also use the `dedicated structure creator app <https://epimodel-graph-editor.vercel.app/>`_,
where you can visualize your model as well.


State Template
--------------
To define the states (compartments) of the model, use the following structure. The `type` of state
(e.g., susceptible, infected) and the `rate` of transition out of the state can be specified.
Substates represent multiple stages within a state.

.. code-block:: json

   {
     "state_data": {
       "state_name": {          // The string used for referencing this state
         "type": "state_type",  // Optional: "susceptible", "infected", "recovered", "dead", etc.
         "n_substates": N,      // Optional: Number of substates (for Erlan states)
         "rate": "rate_name"    // Optional: "Decay" rate
       }
     }
   }

**Example:**

.. code-block:: json

    {
     "state_data": {
       "s": {
          "type": "susceptible"
        },
       "e": {
          "type": "infected",
          "rate": "alpha"
        },
       "i": {
          "type": "infected",
          "rate": "gamma"
        },
       "h": {
          "n_substates": 1,
          "rate": "gamma"
        },
        "r": {
          "type": "recovered"
        }
      }
    }


Transition Template
-------------------
Define how individuals move from one state to another with the transition rules. Transitions may also involve parameters that control how individuals progress between states.

.. code-block:: json

   {
     "trans_data": [
       {
         "source": "source_state",  // The state from which the transition starts
         "target": "target_state",  // The state to which individuals move
         "params": ["param_name"]   // Optional: Parameters controlling the transition (e.g., probabilities, rates)
       }
     ]
   }

Parameters have to match a key in the model_params dictionary.
If a param ends with "_", the value of 1 - param will be used.

**Example:**

.. code-block:: json

    {
      "trans_data": [
        {
          "source": "i",
          "target": "r"
        },
        {
          "source": "e",
          "target": "i",
          "params": [
            "eta_"
          ]
        },
        {
          "source": "e",
          "target": "h",
          "params": [
            "eta"
          ]
        },
        {
          "source": "h",
          "target": "r"
        }
      ]
    }

Transmission Template
---------------------
This section describes how individuals move from the susceptible state to an infected state. Define which
infected states contribute to transmission and any parameters influencing the transmission process.

.. code-block:: json

   {
     "tms_rules": [
       {
         "source": "susceptible_state",      // The susceptible state that will become infected
         "target": "infected_state",         // The state individuals move to after infection
         "actors-params": {                  // Infectious states contributing to transmission
           "infectious_state_1": "param_1",  // Key - name of infectious state, value - relative infectiousness (can be null)
         }
         "susc_params": ["susc"],             // (Optional) Susceptibility parameters
         "inf_params": ["inf"]               // (Optional) Infectivity parameters
       }
     ]
   }

**Example:**

The transmission rule for a simple SEIR model would look like this:

.. code-block:: json

  {
    "tms_rules": [
      {
        "source": "s",
        "target": "e",
        "actors-params": {
          "i": null
          }
      }
    ]
  }


Conclusion
----------

Use these templates as a guide when creating your own configuration files for the EMSA framework.
Customize the states, transitions, and transmission rules to fit the specific dynamics of the
epidemic model you're working on.
