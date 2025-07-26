import numpy as np
from pathlib import Path
import argparse

def load_episode(fn):
    """Loads a single episode from a .npz file."""
    try:
        with fn.open('rb') as f:
            episode = np.load(f)
            # allow_pickle=True is necessary for some numpy versions
            # when loading object arrays.
            episode = {k: episode[k] for k in episode.keys()}
            return episode
    except Exception as e:
        print(f"Error loading episode {fn}: {e}")
        return None

def inspect_buffer(buffer_dir, num_episodes=100, expected_state_dim=15, expected_priv_dim=9):
    """
    Inspects the last N episodes in a replay buffer and prints the shapes
    of the 'state' and 'priv_state' arrays, warning if they are unexpected.

    Args:
        buffer_dir (str or Path): The directory containing the .npz episode files.
        num_episodes (int): The number of recent episodes to inspect.
        expected_state_dim (int): The expected feature dimension for the 'state' array.
        expected_priv_dim (int): The expected feature dimension for the 'priv_state' array.
    """
    buffer_path = Path(buffer_dir)
    if not buffer_path.is_dir():
        print(f"Error: Directory not found at {buffer_path}")
        return

    print(f"Searching for episodes in: {buffer_path.resolve()}")

    # Get all episode files and sort them by timestamp in the filename
    try:
        episode_fns = sorted(buffer_path.glob('*.npz'), key=lambda x: x.stem.split('_')[0], reverse=True)
    except IndexError:
        print("Error: Could not parse filenames to sort by timestamp.")
        print("Ensure filenames are in the format 'timestamp_idx_len.npz'")
        return


    if not episode_fns:
        print("No episode files (.npz) found in the directory.")
        return

    # Get the last N episodes
    recent_episodes = episode_fns[:num_episodes]
    print(f"Found {len(episode_fns)} total episodes. Inspecting the last {len(recent_episodes)}...\n")

    for fn in reversed(recent_episodes): # Reverse to show oldest to newest
        print(f"--- File: {fn.name} ---")
        episode = load_episode(fn)
        if episode is None:
            continue

        # Check state shape
        if 'state' in episode:
            state_shape = episode['state'].shape
            print(f"  'state' shape: {state_shape}")
            if len(state_shape) < 2 or state_shape[1] != expected_state_dim:
                print(f"  !!!! WARNING: 'state' dimension is {state_shape[-1]}, expected {expected_state_dim}.")
        else:
            print("  'state' key not found.")

        # Check priv_state shape
        if 'priv_state' in episode:
            priv_state_shape = episode['priv_state'].shape
            print(f"  'priv_state' shape: {priv_state_shape}")
            if len(priv_state_shape) < 2 or priv_state_shape[1] != expected_priv_dim:
                print(f"  !!!! WARNING: 'priv_state' dimension is {priv_state_shape[-1]}, expected {expected_priv_dim}.")
        else:
            print("  'priv_state' key NOT FOUND.")
        print("-" * (len(fn.name) + 10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect shapes of data in a replay buffer.")
    parser.add_argument('buffer_dir', type=str, help="Path to the replay buffer directory containing .npz files.")
    parser.add_argument('--num', type=int, default=100, help="Number of recent episodes to inspect.")
    parser.add_argument('--state_dim', type=int, default=15, help="Expected dimension of the state vector.")
    parser.add_argument('--priv_dim', type=int, default=9, help="Expected dimension of the privileged state vector.")
    args = parser.parse_args()

    inspect_buffer(args.buffer_dir, args.num, args.state_dim, args.priv_dim)
