import numpy as np
import matplotlib.pyplot as plt
from circuit_simulator import CircuitSimulator
import pickle

def plot_data(x_test, tpoints):
    # Create the figure and the first axis (for Volts)
    _ ,ax1 = plt.subplots(figsize=(10, 6))

    # Plot V_1, V_2, and V_3 on the primary y-axis
    ax1.plot(tpoints, x_test[:, 0], label='$V_1$')
    ax1.plot(tpoints, x_test[:, 1], label='$V_2$', linestyle='--')
    ax1.plot(tpoints, x_test[:, 2], label='$V_3$', linestyle='--')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Volt (V)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a twin axis for I_E (for mA)
    ax2 = ax1.twinx()
    # Scale I_E to mA
    ie_ma = x_test[:, 3] * 1000
    ax2.plot(tpoints, ie_ma, label='$I_E$', color='red', linestyle='--')
    ax2.set_ylabel("Current (mA)")

    # Calculate the ratio of the zero position relative to the range
    # We force the zero to be at the same proportional height on both axes
    def align_zeros(ax_ref, ax_target):
        ymin_ref, ymax_ref = ax_ref.get_ylim()
        rat = ymax_ref / (ymax_ref - ymin_ref)
        ymin_tar, ymax_tar = ax_target.get_ylim()
        if abs(ymin_tar) > abs(ymax_tar):
            new_ymax = ymin_tar * rat / (rat - 1)
            ax_target.set_ylim(ymin_tar, new_ymax)
        else:
            new_ymin = ymax_tar * (rat - 1) / rat
            ax_target.set_ylim(new_ymin, ymax_tar)

    # Apply the alignment
    align_zeros(ax1, ax2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Voltage and Current vs. Time")
    plt.tight_layout()
    plt.show()
def create_dataset(num_samples, amplitude, f, delta_t, T, noise):
    x = [] # To store the simulated transient responses (features)
    y = [] # To store the ground truth R and C values (labels)

    for i in range(num_samples):
        # Randomly sample resistance (1 to 2.5k Ohms) and capacitance (0.1 to 5 microFarads)
        R = np.random.uniform(1.0, 2.5e3)     # Ohms: [1, 2500]
        C = np.random.uniform(0.1e-6, 5e-6)   # Farads: [0.1uF, 5uF]
        # Initialize the Modified Nodal Analysis (MNA) simulator with current parameters
        mna = CircuitSimulator(amplitude, f, R, C)
        y.append([R, C])

        # Initialize state variables 
        x_init = np.zeros((4,))

        # Perform simulation using the Backward Euler method
        transient, _ = mna.BEuler(x_init, delta_t, T, noise = noise)
        x.append(transient)

        # Progress tracking
        if(i % 100 == 0):
            print(f"Created {i+1} samples")

    # Convert lists to NumPy arrays for easier manipulation in ML frameworks
    x = np.array(x)
    y = np.array(y)
    return x, y

def save_dataset(x, y, filename='group_5_dataset.pkl', seed=42,
                 split=(0.70, 0.15, 0.15), standardize=True, log_targets=False):
    """
    Saves a dataset pickle with the structure:

    {
      'X_train': (n_train, T, 4),
      'X_val':   (n_val,   T, 4),
      'X_test':  (n_test,  T, 4),
      'y_train': (n_train, 2),   # [R, C]
      'y_val':   (n_val,   2),
      'y_test':  (n_test,  2),
      # optional:
      'mu': (1, 1, 4), 'sigma': (1, 1, 4), 'indices': {...}, 'target_is_log': bool
    }

    Inputs:
      x: (N, T, 4) time series [V1, V2, V3, IE]
      y: (N, 2)    targets [R, C]
    """
    import pickle
    import numpy as np

    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 3 or x.shape[2] != 4:
        raise ValueError(f"Expected x to have shape (N, T, 4). Got {x.shape}")
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected y to have shape (N, 2). Got {y.shape}")

    N, T, C = x.shape  # C should be 4

    # Optionally log-transform targets
    Y = np.log(y) if log_targets else y

    # Split indices (reproducible)
    train_frac, val_frac, test_frac = split
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(train_frac * N)
    n_val   = int(val_frac * N)
    n_test  = N - n_train - n_val

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    X_train, y_train = x[idx_train], Y[idx_train]
    X_val,   y_val   = x[idx_val],   Y[idx_val]
    X_test,  y_test  = x[idx_test],  Y[idx_test]

    # Standardize per channel using TRAIN stats only (no leakage)
    mu, sigma = None, None
    if standardize:
        # mean/std over samples + time, separately for each channel
        mu = X_train.mean(axis=(0, 1), keepdims=True)      # (1,1,4)
        sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

        X_train = (X_train - mu) / sigma
        X_val   = (X_val   - mu) / sigma
        X_test  = (X_test  - mu) / sigma

    data_to_save = {
        'X_train': X_train,
        'X_val':   X_val,
        'X_test':  X_test,
        'y_train': y_train,
        'y_val':   y_val,
        'y_test':  y_test,
        'mu': mu,
        'sigma': sigma,
        'indices': {'train': idx_train, 'val': idx_val, 'test': idx_test},
        'target_is_log': bool(log_targets),
        'original_shapes': {'X_raw': (N, T, C)}
    }

    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved dataset to {filename}")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    print(f"        y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")
