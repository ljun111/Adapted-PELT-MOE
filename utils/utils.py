import numpy as np
from scipy.spatial.distance import cosine

def standardize_data(data):
    """
    Standardize the data by removing the mean and scaling to unit variance.

    Args:
        data (np.ndarray): The data to be standardized.

    Returns:
        np.ndarray: The standardized data.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-4

    standardized_data = (data - mean) / std

    return standardized_data, mean, std

def cosinesimilarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity

def weightedscores(feature_mean_train, feature_test, PELT_Tsquareds, PELT_SSEsquareds, k):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    weighted_PELT_Tsquareds = []
    weighted_PELT_SSEsquareds = []

    for i in range(len(feature_test[0])):
        corresponding_Tsquareds = np.array([PELT_Tsquareds[j][i] for j in range(len(feature_test))])
        corresponding_SSEsquareds = np.array([PELT_SSEsquareds[j][i] for j in range(len(feature_test))])
        corresponding_feature = np.array([feature_test[j][i] for j in range(len(feature_test))])
        corresponding_feature = np.squeeze(corresponding_feature)
        # print("corresponding_Tsquareds:",corresponding_Tsquareds.shape)
        # print("corresponding_feature:", corresponding_feature.shape)
        cosines = np.zeros(len(feature_test))
        for j in range(len(corresponding_feature)):
            cosines[j] = cosinesimilarity(corresponding_feature[j,:],feature_mean_train[j])

        weights = softmax(cosines)
        # print("weights:",weights)
        top_k_indices = np.argsort(weights)[-k:]
        # print("top_k_indices:", top_k_indices)
        top_k_weights = weights[top_k_indices]
        # print("top_k_weights:", top_k_weights)
        normalized_weights = top_k_weights / np.sum(top_k_weights)
        # print("normalized_weights:", normalized_weights)
        weighted_PELT_T = np.sum(normalized_weights * corresponding_Tsquareds[top_k_indices])
        weighted_SSE = np.sum(normalized_weights * corresponding_SSEsquareds[top_k_indices])

        weighted_PELT_Tsquareds.append(weighted_PELT_T)
        weighted_PELT_SSEsquareds.append(weighted_SSE)
    print("weighted_PELT_Tsquareds:",len(weighted_PELT_Tsquareds))
    print("weighted_PELT_SSEsquareds:", len(weighted_PELT_SSEsquareds))
    return weighted_PELT_Tsquareds, weighted_PELT_SSEsquareds

def reconsimilarity(v1, v2):
    recon = np.linalg.norm(v1 - v2)**2
    return recon

def weightedscores_recon(feature_mean_train, feature_test, PELT_Tsquareds, PELT_SSEsquareds, k):
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    weighted_PELT_Tsquareds = []
    weighted_PELT_SSEsquareds = []

    for i in range(len(feature_test[0])):
        corresponding_Tsquareds = np.array([PELT_Tsquareds[j][i] for j in range(len(feature_test))])
        corresponding_SSEsquareds = np.array([PELT_SSEsquareds[j][i] for j in range(len(feature_test))])
        corresponding_feature = np.array([feature_test[j][i] for j in range(len(feature_test))])
        corresponding_feature = np.squeeze(corresponding_feature)
        # print("corresponding_Tsquareds:",corresponding_Tsquareds.shape)
        # print("corresponding_feature:", corresponding_feature.shape)
        recons = np.zeros(len(feature_test))
        for j in range(len(corresponding_feature)):
            recons[j] = reconsimilarity(corresponding_feature[j,:],feature_mean_train[j])
        # print("recons:",recons)
        recons = 1/recons
        weights = recons / np.sum(recons)
        # print("weights:",weights)
        top_k_indices = np.argsort(weights)[-k:]
        # print("top_k_indices:", top_k_indices)
        top_k_weights = weights[top_k_indices]
        # print("top_k_weights:", top_k_weights)
        normalized_weights = top_k_weights / np.sum(top_k_weights)
        # print("normalized_weights:", normalized_weights)
        weighted_PELT_T = np.sum(normalized_weights * corresponding_Tsquareds[top_k_indices])
        weighted_SSE = np.sum(normalized_weights * corresponding_SSEsquareds[top_k_indices])

        weighted_PELT_Tsquareds.append(weighted_PELT_T)
        weighted_PELT_SSEsquareds.append(weighted_SSE)
    print("weighted_PELT_Tsquareds:",len(weighted_PELT_Tsquareds))
    print("weighted_PELT_SSEsquareds:", len(weighted_PELT_SSEsquareds))
    return weighted_PELT_Tsquareds, weighted_PELT_SSEsquareds

def min_sse_t2(feature_test, PELT_Tsquareds, PELT_SSEsquareds):
    min_SSE_Tsquareds = []

    for i in range(len(feature_test[0])):
        corresponding_Tsquareds = np.array([PELT_Tsquareds[j][i] for j in range(len(feature_test))])
        corresponding_SSEsquareds = np.array([PELT_SSEsquareds[j][i] for j in range(len(feature_test))])
        # print("corresponding_SSEsquareds:",corresponding_SSEsquareds)
        min_sse_t2_indice = corresponding_SSEsquareds.argmin()
        # print("min_sse_t2_indice:", min_sse_t2_indice)
        min_SSE_Tsquareds.append(corresponding_Tsquareds[min_sse_t2_indice])
    print("min_SSE_Tsquareds:",len(min_SSE_Tsquareds))
    return min_SSE_Tsquareds


def downsample(data, labels=None, down_size=10):
    """
    Perform downsampling on data and labels

    Args:
        data: Input data (n_samples, ...)
        labels: Optional labels (n_samples,)
        window_size: Downsampling window size (default=10)

    Returns:
        Downsampled data and labels (if labels were provided)
    """
    n = len(data) // down_size * down_size

    downsampled_data = np.mean(
        data[:n].reshape(-1, down_size, *data.shape[1:]),
        axis=1
    )

    if labels is not None:
        downsampled_labels = np.max(
            labels[:n].reshape(-1, down_size),
            axis=1
        ).astype(int)
        return downsampled_data, downsampled_labels

    return downsampled_data