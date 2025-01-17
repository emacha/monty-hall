import numpy as np

options = ("goat", "goat", "car")
def standard_monty_round(options: tuple):
    # shuffle the doors
    doors = np.random.permutation(options)
    # We always pick the first door, doesn't matter for the simulation
    initial_pick, rest = doors[0], doors[1:]
    # The host opens any door that's not a car
    open_index = np.where(rest != "car")[0][0]
    open_door = rest[open_index]
    # Remove the open door. We are left with one closed door. That's the switch pick
    switch_pick = np.delete(rest, open_index)
    return np.hstack((initial_pick, switch_pick, open_door))


def run_standard_simulation(options: tuple, n_simulations: int):
    samples = np.vstack([standard_monty_round(options) for _ in range(n_simulations)])
    probabilities = (samples == "car").mean(axis=0)
    print(f"Initial pick: {probabilities[0]:.2%}, Switch pick: {probabilities[1]:.2%}")    

def calculate_probabilities(sample_counts) -> str:
    return ", ".join([f"{value}: {count / sample_counts.counts.sum():.2%}" for value, count in zip(*sample_counts)])


run_standard_simulation(100000)



# options = ("goat", "goat", "car")
options = ("pet goat", "goat", "car")
n_simulations = 100000
samples = np.vstack([standard_monty_round(options) for _ in range(n_simulations)])
samples_where_open_door_is_goat = samples[np.where(samples[:, 2] == "goat")]

initial_picks = samples_where_open_door_is_goat[:, 0]
switch_picks = samples_where_open_door_is_goat[:, 1]
print("Initial pick probabilities: " + calculate_probabilities(np.unique_counts(initial_picks)))
print("Switch pick probabilities:  " + calculate_probabilities(np.unique_counts(switch_picks)))
