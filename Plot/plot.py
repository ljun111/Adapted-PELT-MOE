import matplotlib.pyplot as plt
import numpy as np

def plot_data_and_reconstructions(original_data, reconstructed_data, changepoints, filename):
    """
    Plot the original and reconstructed data with vertical dashed lines at changepoints
    and save the figure to a file.

    Args:
        original_data (np.ndarray): The original data as a 2D array.
        reconstructed_data (np.ndarray): The reconstructed data as a 2D array.
        changepoints (list of int): The indices of the changepoints.
        filename (str): The filename to save the figure.
    """
    # original_data = original_data[label == 1]
    # reconstructed_data = reconstructed_data[label == 1]
    # reconstructed_data_onemodel = reconstructed_data_onemodel[label == 1]

    num_features = original_data.shape[1]
    num_subplots = num_features
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))

    if num_features == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(original_data[:, i], label='Original', color='blue')
        ax.plot(reconstructed_data[:, i], label='Reconstructed', color='red')
        # ax.plot(reconstructed_data_onemodel[:, i], label='Reconstructed_Onemodel', color='yellow')

        for cp in changepoints:
            ax.axvline(x=cp, color='green', linestyle='--', linewidth=1)

        ax.set_title(f'Feature {i+1}')
        ax.set_ylim((-1, 1))
        ax.legend()

    plt.tight_layout()
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.show()


def plot_loss(epoch_losses):
    """Plot the loss over epochs.

    Args:
        epoch_losses (list): List of average loss values for each epoch.
    """
    epochs = range(1, len(epoch_losses)+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

def plot_feature(feature):
    """Plot the feature.

    Args:
        feature (np.ndarray): The feature data as a 2D array.
    """
    n, p = feature.shape
    fig, axs = plt.subplots(1, p, figsize=(p * 5, 4))
    fig.suptitle('Columns Data Points')
    if p == 1:
        axs = [axs]
    for i in range(p):
        axs[i].plot(feature[:, i], label=f'Column {i + 1}')
        axs[i].set_title(f'Column {i + 1} Data Points')
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')
        axs[i].legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_t2(t2_values, changepoints, labels):
    """Plot the T2 statistical values with red points at indices where labels are 1.

    Args:
        t2_values (np.ndarray): The T2 statistical data as a 1D array.
        changepoints (list): List of changepoint indices.
        labels (np.ndarray): Array of labels where 1 indicates an anomaly.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t2_values, label='T2 Statistic')

    anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
    anomaly_t2_values = [t2_values[idx] for idx in anomaly_indices]
    ax.scatter(anomaly_indices, anomaly_t2_values, color='red', label='Anomaly')

    for cp in changepoints:
        ax.axvline(x=cp, color='green', linestyle='--', linewidth=1)

    ax.set_title('T2 Statistical Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('T2 Value')
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_sse(sse_values, changepoints, labels):
    """Plot the T2 statistical values with red points at indices where labels are 1.

    Args:
        sse_values (np.ndarray): The SSE statistical data as a 1D array.
        changepoints (list): List of changepoint indices.
        labels (np.ndarray): Array of labels where 1 indicates an anomaly.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sse_values, label='SSE Statistic')

    anomaly_indices = [i for i, label in enumerate(labels) if label == 1]
    anomaly_t2_values = [sse_values[idx] for idx in anomaly_indices]
    ax.scatter(anomaly_indices, anomaly_t2_values, color='red', label='Anomaly')

    for cp in changepoints:
        ax.axvline(x=cp, color='green', linestyle='--', linewidth=1)

    ax.set_title('SSE Statistical Values')
    ax.set_xlabel('Index')
    ax.set_ylabel('SSE Value')
    # ax.set_ylim((0,50))
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_reconstruction_error(reconstruction_errors, labels):
    """Plot the reconstruction errors in two subplots. The first subplot includes red vertical lines at indices where labels are 1.
    The second subplot only includes points where labels are 0.

    Args:
        reconstruction_errors (list): The reconstruction error values as a list.
        labels (np.ndarray): Array of labels where 1 indicates an anomaly.
    """
    reconstruction_errors = np.array(reconstruction_errors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))

    anomaly_indices = np.where(labels == 1)[0]
    for idx in anomaly_indices:
        ax1.axvline(x=idx, color='lightcoral', linewidth=2)
    ax1.plot(reconstruction_errors, label='Reconstruction Error')
    ax1.set_title('Reconstruction Error with Anomalies Highlighted')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Reconstruction Error')
    ax1.legend()

    normal_indices = [i for i, label in enumerate(labels) if label == 0]
    reconstruction_errors_normal = [reconstruction_errors[idx] for idx in normal_indices]
    ax2.plot(normal_indices, reconstruction_errors_normal, 'bo', label='Normal Points')
    ax2.set_title('Reconstruction Error for Normal Points')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Reconstruction Error')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_encoders(data_array):
    """
    Plot a graph with n lines, where n is the number of rows in the input 2D array.
    Each line represents the p data points of that row.

    Args:
        data_array (np.ndarray): A 2D array with shape (n, p).
    """
    n, p = data_array.shape
    x = np.arange(p)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n):
        ax.plot(x, data_array[i, :], label=f'Encoder {i + 1}')

    ax.set_title('Data Plot')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_candidates(changepoints, candidate_score, candidate_recon_loss, candidate_likelihood_value, candidate_cos_sim, changepoint_type, data_test_label):
    fig, ax = plt.subplots(figsize=(10, 6))

    anomaly_added = False
    for idx, label in enumerate(data_test_label):
        if label == 1:
            ax.axvline(x=idx, color='lightgray', linestyle='-', alpha=1, linewidth=1,
                      label='Anomaly' if not anomaly_added else "")
            anomaly_added = True

    ax.plot(candidate_score, label='Anomaly Score', color='blue')
    # ax.plot(candidate_recon_loss, label='Candidate Reconstruction Loss', color='green')
    # ax.plot(candidate_likelihood_value, label='Candidate Likelihood Value', color='yellow')
    # ax.plot(candidate_cos_sim, label='Candidate Cosine Similarity', color='purple')

    condition_label_added = False
    for cp in changepoints:
        if changepoint_type[cp] == 'condition':
            if not condition_label_added:
                ax.axvline(x=cp, color='orange', linestyle='--', label='Condition Changepoint')
                condition_label_added = True
            else:
                ax.axvline(x=cp, color='orange', linestyle='--')

    # ax.set_ylim((0,1))
    ax.legend(loc='upper left')

    # ax.set_title('Candidate Metrics and Condition Changepoints')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')

    plt.show()

def plot_predictions_with_mismatches(y_pred, data_test_label):
    if len(y_pred) != len(data_test_label):
        raise ValueError("y_pred 和 data_test_label 的长度必须一致")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(y_pred)):
        if data_test_label[i] == 1:
            ax.axvline(x=i, color='lightgray', linestyle='-', alpha=1, label='Mismatch' if i == 0 else "")

    ax.plot(y_pred, label='y_pred (Predictions)', color='blue', marker='o')
    # ax.plot(data_test_label, label='data_test_label (True Labels)', color='green', marker='x')

    ax.legend()
    ax.set_title('Predictions vs True Labels with Mismatches Highlighted')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    plt.show()

def plot_data_with_changepoints(original_data, changepoints, changepoints_type):
    num_features = original_data.shape[1]
    num_subplots = num_features
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))

    for i, ax in enumerate(axes):
        ax.plot(original_data[:, i], label='Original', color='blue')

        condition_label_added = False
        weight_label_added = False
        for cp in changepoints:
            if changepoints_type[cp] == 'condition':
                if not condition_label_added:
                    ax.axvline(x=cp, color='orange', linestyle='--', alpha=0.7, label='Condition Changepoint')
                    condition_label_added = True
                else:
                    ax.axvline(x=cp, color='orange', linestyle='--', alpha=0.7)
            if changepoints_type[cp] == 'weight':
                if not weight_label_added:
                    ax.axvline(x=cp, color='green', linestyle='--', alpha=0.7, label='Weight Changepoint')
                    weight_label_added = True
                else:
                    ax.axvline(x=cp, color='green', linestyle='--', alpha=0.7)

        ax.set_title(f'Feature {i+1}')
        ax.legend()

    plt.tight_layout()
    plt.title('Training Data with Condition Changepoints Highlighted')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.show()

def plot_reconstruction_data_with_changepoints(original_data, reconstruction_data, changepoints, changepoints_type):
    num_features = original_data.shape[1]
    num_subplots = num_features
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 5 * num_subplots))

    for i, ax in enumerate(axes):
        ax.plot(original_data[:, i], label='Original', color='blue')
        ax.plot(reconstruction_data[:, i], label='Reconstruction', color='red')

        condition_label_added = False
        weight_label_added = False
        for cp in changepoints:
            if changepoints_type[cp] == 'condition':
                if not condition_label_added:
                    ax.axvline(x=cp, color='orange', linestyle='--', alpha=0.7, label='Condition Changepoint')
                    condition_label_added = True
                else:
                    ax.axvline(x=cp, color='orange', linestyle='--', alpha=0.7)
            if changepoints_type[cp] == 'weight':
                if not weight_label_added:
                    ax.axvline(x=cp, color='green', linestyle='--', alpha=0.7, label='Weight Changepoint')
                    weight_label_added = True
                else:
                    ax.axvline(x=cp, color='green', linestyle='--', alpha=0.7)

        ax.set_title(f'Feature {i+1}')
        ax.legend()

    plt.tight_layout()
    plt.title('Training Data with Condition Changepoints Highlighted')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.show()

def plot_data_with_truechangepoints(data_test_tensor_array, changepoints, true_changepoints):
    changepoints = changepoints[1:]
    num_time_steps, num_features = data_test_tensor_array.shape
    time_axis = np.arange(num_time_steps)
    plt.figure(figsize=(10, 6))

    for i in range(num_features):
        plt.plot(time_axis, data_test_tensor_array[:, i], label=f'Feature {i + 1}')

    for cp in changepoints:
        plt.axvline(x=cp, color='red', linestyle='--', label='Detected Changepoint' if cp == changepoints[0] else "")

    colors = ['lightcoral', 'lightblue', 'lightgreen', 'moccasin', 'thistle', 'wheat', 'lightpink', 'lightgray',
              'palegoldenrod', 'paleturquoise']

    start_idx = 0
    for idx, tcp in enumerate(true_changepoints):
        plt.axvspan(start_idx, tcp, color=colors[idx % len(colors)], alpha=0.2)
        start_idx = tcp

    plt.axvspan(start_idx, num_time_steps, color=colors[len(true_changepoints) % len(colors)], alpha=0.2)

    plt.legend(loc='upper right')

    # plt.title('Data with Changepoints')
    plt.xlabel('Index')
    plt.ylabel('Value')

    plt.show()


def plot_features_with_anomalies(data_test, test_label):
    """
    Plot subplots for each variable in data_test with anomaly points (test_label=1) marked.

    Args:
        data_test: 2D array (n_samples, n_features)
        test_label: 1D array (n_samples,), where 1 indicates anomaly points
    """
    n_features = data_test.shape[1]
    n_samples = data_test.shape[0]

    fig, axes = plt.subplots(n_features, 1, figsize=(10, 2 * n_features))

    if n_features == 1:
        axes = [axes]

    x = np.arange(n_samples)

    for i in range(n_features):
        ax = axes[i]
        feature_data = data_test[:, i]

        ax.plot(x, feature_data, 'o', color='gray', markersize=3, alpha=0.5, label='Normal')

        anomaly_idx = np.where(test_label == 1)[0]

        ax.plot(anomaly_idx, feature_data[anomaly_idx], 'o',
                color='red', markersize=4, label='Anomaly')

        ax.set_title(f'Feature {i + 1}')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Value')

        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()