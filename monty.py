# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
# ]
# ///
import numpy as np
from numpy.lib._arraysetops_impl import UniqueCountsResult

def monty_round(options: tuple) -> np.ndarray:
    """Run a round of the Monty Hall problem, given a tuple of animals behind the doors."""
    # shuffle the doors
    doors = np.random.permutation(options)
    # We always pick the first door, doesn't matter for the simulation
    initial_pick, rest = doors[0], doors[1:]
    # The host opens any door that's not a car
    open_index = np.where(rest != "car")[0][0]
    open_door = rest[open_index]
    # Remove the open door. We are left with one closed door. That's the switch pick
    switch_pick = np.delete(rest, open_index)[0]
    return np.hstack((initial_pick, switch_pick, open_door))


def calculate_probabilities(sample_counts: UniqueCountsResult) -> str:
    """Calculate the probabilities of each value in the sample_counts. Return a string for easy printing."""
    return ", ".join([f"{value}: {count / sample_counts.counts.sum():.1%}" for value, count in zip(*sample_counts)])


def run_simulation(options: tuple, n_simulations: int):
    """Run a simulation of the Monty Hall problem and print the results."""
    print(f"\nRunning {n_simulations} simulations with the problem: {options}")
    samples = np.vstack([monty_round(options) for _ in range(n_simulations)])
    print("\nOverall probabilities")
    initial_picks = samples[:, 0]
    switch_picks = samples[:, 1]
    print("Initial pick probabilities: " + calculate_probabilities(np.unique_counts(initial_picks)))
    print("Switch pick probabilities:  " + calculate_probabilities(np.unique_counts(switch_picks)))

    initial_picks = samples[:, 0]
    switch_picks = samples[:, 1]
    possible_open_doors = np.unique(samples[:, 2])
    for open_door in possible_open_doors:
        print(f"\nOpen door: {open_door}")
        samples_open_door = samples[np.where(samples[:, 2] == open_door)]
        initial_picks = samples_open_door[:, 0]
        switch_picks = samples_open_door[:, 1]
        print("Initial pick probabilities: " + calculate_probabilities(np.unique_counts(initial_picks)))
        print("Switch pick probabilities:  " + calculate_probabilities(np.unique_counts(switch_picks)))

if __name__ == "__main__":
    n_simulations = 100000
    run_simulation(("goat", "goat", "car"), n_simulations)
    run_simulation(("pet goat", "goat", "car"), n_simulations)
