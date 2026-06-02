import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values = np.asarray(values, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    num_states = values.shape[0]
    new_values = np.zeros(num_states)

    for s in range(num_states): 
        action_values = []

        for a in range(transitions.shape[1]):
            expected_value = np.sum(transitions[s, a]*values)
            q = rewards[s, a] + gamma * expected_value 
            action_values.append(q) 

        new_values[s] = np.max(action_values)
    return list(new_values)
    