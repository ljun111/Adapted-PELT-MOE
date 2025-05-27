import torch
import numpy as np
import torch.nn as nn
import os
from Online_PELT_MOE_separated.Encoder_train import train_encoder_decoder,reshape_to_3d,normalize_tensor,update_models_with_online_gradient_descent, weighted_reconstruction
from Online_PELT_MOE_separated.candidate_detection import compute_scores_for_3d_input
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parallel_estimate_reconstruction_error(model_name, data, seg_len, cps, epochs, learning_rate, latent_space_size,
                                           hidden_dim, batch_size, window_size, n_components):
    """Estimate reconstruction error using Encoder-Decoder model for each segment."""
    reconstruction_errors = []
    for cp in cps:
        data_seg = data[cp:seg_len, :]
        data_seg = data_seg.to(dtype=torch.float32)
        data_seg_3d = reshape_to_3d(data_seg, window_size)

        model_errors = []
        for model in model_name:
            model = [model]
            results = train_encoder_decoder(
                model, data_seg, epochs, learning_rate,
                latent_space_size, hidden_dim, batch_size, window_size, n_components
            )

            encoder = results[model[0]]['encoder']
            decoder = results[model[0]]['decoder']
            with torch.no_grad():
                first_time_step_data = data_seg_3d[:, -1:, :]
                first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
                first_time_step_data = first_time_step_data.view(data_seg_3d.shape[0], -1)

                latent_space = encoder(first_time_step_data)
                reconstruction = decoder(latent_space)

                reconstruction_error = MSELoss(reduction='sum')(reconstruction, first_time_step_data) / window_size
                model_errors.append(reconstruction_error.item())

        reconstruction_errors.append(np.min(model_errors))
    print("reconstruction_errors:",reconstruction_errors)
    return reconstruction_errors


def parallel_estimate_weighted_reconstruction_error(recon_experts, data, seg_len, cps, weights, window_size, model_name):
    """
    Estimate reconstruction error for each data segment using weighted expert models.
    """
    print("weights:",weights)
    reconstruction_errors = []
    for cp in cps:
        data_seg = data[cp:seg_len, :]
        data_seg = data_seg.to(dtype=torch.float32)
        data_seg_3d = reshape_to_3d(data_seg, window_size)

        first_time_step_data = data_seg_3d[:, -1:, :]
        first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
        first_time_step_data = first_time_step_data.view(data_seg_3d.shape[0], -1)

        weighted_reconstruction = torch.zeros_like(first_time_step_data)

        weight_idx = 0
        for model_experts in recon_experts:  # model_experts is [(Encoder, Decoder), ...]
            for encoder, decoder in model_experts:  # Directly unpack tuple
                latent = encoder(first_time_step_data)
                expert_reconstruction = decoder(latent)
                weighted_reconstruction += weights[weight_idx] * expert_reconstruction
                weight_idx += 1
        total_weight = sum(weights)
        weighted_reconstruction /= total_weight
        reconstruction_error = MSELoss(reduction='sum')(weighted_reconstruction, first_time_step_data) / window_size
        reconstruction_errors.append(reconstruction_error.item())
    print("reconstruction_errors_weight:",reconstruction_errors)
    return reconstruction_errors


def parallel_estimate_weighted_reconstruction_error_with_training(recon_experts, data, seg_len, cps, weights, window_size, epochs, learning_rate, batch_size,model_name):
    """
    Estimate reconstruction error for each data segment using weighted expert models and update weights through gradient descent.
    """
    reconstruction_errors = []
    num_experts_per_model = len(recon_experts[0]) if recon_experts else 0
    weights = torch.tensor(weights, dtype=torch.float32, requires_grad=True)
    optimizer = Adam([weights], lr=learning_rate)

    if len(weights) != len(model_name):
        print("weight_changed_started!!!")
        for epoch in range(epochs):
            for cp in cps:
                data_seg = data[cp:seg_len, :]
                data_seg = data_seg.to(dtype=torch.float32)
                data_seg_3d = reshape_to_3d(data_seg, window_size)
                # data_seg_3d = normalize_tensor(data_seg_3d)
                first_time_step_data = data_seg_3d[:, -1:, :]
                first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
                first_time_step_data = first_time_step_data.view(data_seg_3d.shape[0], -1)
                weighted_reconstruction = torch.zeros_like(first_time_step_data)

                weight_idx = 0
                for model_experts in recon_experts:  # model_experts is [(Encoder, Decoder), ...]
                    for encoder, decoder in model_experts:  # Directly unpack tuple
                        latent = encoder(first_time_step_data)
                        expert_reconstruction = decoder(latent)
                        weighted_reconstruction += weights[weight_idx] * expert_reconstruction
                        weight_idx += 1
                total_weight = sum(weights)
                weighted_reconstruction /= total_weight

                reconstruction_error = MSELoss(reduction='sum')(weighted_reconstruction,
                                                                    first_time_step_data) / window_size

                if epoch == epochs - 1:
                    reconstruction_errors.append(reconstruction_error.item())

                optimizer.zero_grad()
                reconstruction_error.backward()
                optimizer.step()

                # print("weights.data:",weights.data)
                with torch.no_grad():
                    weights.data = torch.clamp(weights.data, min=0)
                    weights.data = weights.data / weights.data.sum()

    else:
        weights.data = torch.tensor([1.0 if name == 'Encoder_base' else 0.0 for name in model_name], dtype=torch.float32)
        for cp in cps:
            data_seg = data[cp:seg_len, :]
            data_seg = data_seg.to(dtype=torch.float32)
            data_seg_3d = reshape_to_3d(data_seg, window_size)
            # data_seg_3d = normalize_tensor(data_seg_3d)
            first_time_step_data = data_seg_3d[:, -1:, :]
            first_time_step_data = first_time_step_data.repeat(1, window_size, 1)
            first_time_step_data = first_time_step_data.view(data_seg_3d.shape[0], -1)
            weighted_reconstruction = torch.zeros_like(first_time_step_data)

            model_errors = []
            weight_idx = 0
            for model_experts in recon_experts:  # model_experts is [(Encoder, Decoder), ...]
                for encoder, decoder in model_experts:  # Directly unpack tuple
                    latent = encoder(first_time_step_data)
                    expert_reconstruction = decoder(latent)
                    weighted_reconstruction += weights[weight_idx] * expert_reconstruction
                    weight_idx += 1
                    reconstruction_error = MSELoss(reduction='sum')(weighted_reconstruction, first_time_step_data)/window_size
                    model_errors.append(reconstruction_error.item())
            reconstruction_errors.append(np.min(model_errors))
            # reconstruction_errors.append(float('inf'))
    weights_results = weights.detach().numpy()
    print("weights_after_changed:",weights_results)
    # print("weights_after_changed:", round(weights_results[1], 10))
    print("reconstruction_errors_weight_changed:", reconstruction_errors)
    return reconstruction_errors, weights.detach().numpy()


def segment_online(model_name, data, min_seg_len, K, beta, gamma, epochs, learning_rate, batch_size, latent_space_size, hidden_dim, window_size, recon_experts, feature_experts, feature_means, feature_covariances, train_data_means, tau, lambda1, lambda2, lambda3, n_components):
    """Perform segmentation using PELT algorithm with reconstruction error.
     """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(2035)

    length = data.shape[0]
    score = torch.full((length, length), float('inf'), device=data.device)
    min_score = torch.zeros(length, device=data.device)
    valid_cp = [0]
    changepoints = [0]
    changepoint_type = ['weight'] * length
    y_pred = [int(0)] * len(data)
    anomaly_candidate = []
    candidate_score = np.array([0])
    candidate_recon_loss = []
    candidate_likelihood_value = []
    candidate_cos_sim = []

    seg_len = 1
    weights = [0] * (len(model_name))
    weights[0] = 1.0
    weights_temp = weights
    while seg_len < length:
        if max(changepoints) != 0:
            changepoint_type[changepoints[1]] = 'condition'
        # beta_recon = beta + 10*np.log(len(recon_experts))
        # gamma_recon = gamma / (1 + 10*np.log(len(recon_experts)))

        label_index = seg_len - 1
        data_now = data[label_index:(label_index+1), :]
        data_now_3d = reshape_to_3d(data_now, window_size)
        # data_now_3d = normalize_tensor(data_now_3d)
        score_data_seg_now, recon_losses_now = compute_scores_for_3d_input(data_now_3d, recon_experts, feature_experts,
                                                     feature_means, feature_covariances, train_data_means,
                                                     weights_temp, lambda1, lambda2, lambda3)
        candidate_score = np.append(candidate_score,score_data_seg_now)
        candidate_recon_loss.append(recon_losses_now)
        print("score_data_seg_now:",score_data_seg_now)

        for i in range(len(score_data_seg_now)):
            if score_data_seg_now[i] > tau:
                anomaly_candidate.append(label_index)

        segment_score = torch.full((length,), float('inf'), device=data.device)
        segment_score_condition = torch.full((length,), float('inf'), device=data.device)
        segment_score_weight = torch.full((length,), float('inf'), device=data.device)
        segment_score_weight_changed = torch.full((length,), float('inf'), device=data.device)
        seg_cp = [cp for cp in valid_cp if cp < (seg_len - min_seg_len)]
        print("seg_len:",seg_len)
        print("valid_cp:", valid_cp)
        print("seg_cp_before_append:",seg_cp)

        if len(seg_cp) == 0:
            min_score[seg_len] = float('inf')
            valid_cp.append(seg_len)
            seg_len += 1
            continue

        if max(changepoints) != 0:
            if max(changepoints) not in valid_cp:
                valid_cp.insert(0, max(changepoints))
            valid_cp = [x for x in valid_cp if x != 0 and x>= max(changepoints)]
            if max(changepoints) not in seg_cp:
                seg_cp.insert(0, max(changepoints))
            seg_cp = [x for x in seg_cp if x != 0 and x>= max(changepoints)]

        print("seg_cp_after_append:",seg_cp)

        reconstruction_errors = [float('inf')] * length
        reconstruction_errors_weight = [float('inf')] * length
        reconstruction_errors_weight_changed = [float('inf')] * length
        should_compute_errors = any((cp - max(changepoints)) > min_seg_len for cp in seg_cp)

        if should_compute_errors or len(changepoints) == 1:
            all_reconstruction_errors = parallel_estimate_reconstruction_error(model_name, data, seg_len, seg_cp, epochs, learning_rate,
                                                                               latent_space_size, hidden_dim, batch_size, window_size, n_components)
            all_reconstruction_errors_weight = parallel_estimate_weighted_reconstruction_error(recon_experts, data, seg_len, seg_cp, weights_temp, window_size, model_name)
            all_reconstruction_errors_weight_changed, changed_weights = parallel_estimate_weighted_reconstruction_error_with_training(recon_experts, data, seg_len, seg_cp, weights_temp, window_size, epochs, learning_rate, batch_size, model_name)

            if max(changepoints) != 0:
                if changepoint_type[max(changepoints)] == 'condition':
                    all_reconstruction_errors_weight[0] = all_reconstruction_errors[0]
                    all_reconstruction_errors_weight_changed[0] = all_reconstruction_errors[0]
                else:
                    all_reconstruction_errors[0] = all_reconstruction_errors_weight_changed[0]
                    all_reconstruction_errors_weight[0] = all_reconstruction_errors_weight_changed[0]

            for i, cp in enumerate(seg_cp):
                if (seg_len - cp) >= min_seg_len:
                    reconstruction_errors[i] = all_reconstruction_errors[i]
                    reconstruction_errors_weight[i] = all_reconstruction_errors_weight[i]
                    reconstruction_errors_weight_changed[i] = all_reconstruction_errors_weight_changed[i]
        else:
            if max(changepoints) != 0:
                max_changepoint = [max(changepoints)]
                max_changepoint_type = changepoint_type[max(changepoints)]
                reconstruction_errors[0] = parallel_estimate_reconstruction_error(model_name, data, seg_len, max_changepoint, epochs, learning_rate,
                                                                               latent_space_size, hidden_dim, batch_size, window_size, n_components)[0]
                reconstruction_errors_weight[0] = parallel_estimate_weighted_reconstruction_error(recon_experts, data, seg_len, max_changepoint, weights_temp, window_size, model_name)[0]
                reconstruction_errors_weight_changed_list,_ = parallel_estimate_weighted_reconstruction_error_with_training(recon_experts, data, seg_len, max_changepoint, weights_temp, window_size, epochs, learning_rate, batch_size, model_name)
                reconstruction_errors_weight_changed[0] = reconstruction_errors_weight_changed_list[0]
                if max_changepoint_type == 'condition':
                    reconstruction_errors_weight[0] = reconstruction_errors[0]
                    reconstruction_errors_weight_changed[0] = reconstruction_errors[0]
                else:
                    reconstruction_errors[0] = reconstruction_errors_weight_changed[0]
                    reconstruction_errors_weight[0] = reconstruction_errors_weight_changed[0]

        for i, cp in enumerate(seg_cp):
            if (seg_len - cp) < min_seg_len:
                segment_score_condition[cp] = float('inf')
                segment_score_weight[cp] = float('inf')
                segment_score_weight_changed[cp] = float('inf')
            else:
                if i == 0:
                    segment_score_condition[cp] = reconstruction_errors[0]
                    segment_score_weight[cp] = reconstruction_errors_weight[0]
                    segment_score_weight_changed[cp] = reconstruction_errors_weight_changed[0]
                    min_score_temp = torch.min(segment_score_condition[cp], segment_score_weight[cp])
                    segment_score[cp] = torch.min(min_score_temp,segment_score_weight_changed[cp])
                else:
                    l_seg_score = min_score[cp]
                    r_seg_score = reconstruction_errors[i]
                    segment_score_condition[cp] = l_seg_score + r_seg_score + beta
                    r_seg_score_weight = reconstruction_errors_weight[i]
                    segment_score_weight[cp] = l_seg_score + r_seg_score_weight
                    r_seg_score_weight_changed = reconstruction_errors_weight_changed[i]
                    segment_score_weight_changed[cp] = l_seg_score + r_seg_score_weight_changed + gamma

                    min_score_temp = torch.min(segment_score_condition[cp], segment_score_weight[cp])
                    segment_score[cp] = torch.min(min_score_temp, segment_score_weight_changed[cp])
                    if segment_score_condition[cp] <= segment_score_weight[cp] and segment_score_condition[cp] <= segment_score_weight_changed[cp]:
                        changepoint_type[cp] = "condition"
                    if segment_score_weight_changed[cp] <= segment_score_condition[cp] and segment_score_weight_changed[cp] <= segment_score_weight[cp]:
                        changepoint_type[cp] = "weight"
                    # print("segment_score:",segment_score[cp])
                    print("min_score:", min_score[cp])
                    # print("segment_score_condition:", segment_score_condition[cp])
                    # print("segment_score_weight:", segment_score_weight[cp])
                    # print("segment_score_weight_changed:", segment_score_weight_changed[cp])
                # print("reconstruction_errors:",reconstruction_errors[i])

        score[seg_len, :len(segment_score)] = segment_score
        min_score[seg_len] = segment_score.min()
        print("score:",score[seg_len, :len(segment_score)])
        print("min_score:",min_score)
        non_inf_mask = ~torch.isinf(score[seg_len, :len(segment_score)])
        non_inf_values = score[seg_len, :len(segment_score)][non_inf_mask]
        if len(non_inf_values) > 1:
            differences = np.abs(non_inf_values[1:].detach().cpu().numpy() - non_inf_values[0].detach().cpu().numpy())
            above_threshold_mask = differences < 1.0

            if above_threshold_mask.any():
                non_inf_values[1:][above_threshold_mask] += 1

            score[seg_len, :len(segment_score)][non_inf_mask] = non_inf_values

        for cp in seg_cp:
            if score[seg_len, cp] - K >= min_score[seg_len]:
                valid_cp.remove(cp)

        valid_cp.append(seg_len)

        if seg_len > min_seg_len:
            curr_cp = np.argmin(score[seg_len].cpu().numpy())
            changepoints_type_currup = True
            # if len(recon_experts) == 1:
            #     if changepoint_type[curr_cp] == 'weight':
            #         changepoints_type_currup = False
            if curr_cp not in changepoints and curr_cp - max(changepoints) > min_seg_len and changepoints_type_currup:
                min_score[curr_cp] = 0
                min_score[(curr_cp+1):(curr_cp+min_seg_len+2)] = float('inf')

                if len(weights_temp) == len(model_name):
                    changepoint_type[curr_cp] = 'condition'
                anomaly_candidate = []
                changepoints.append(curr_cp)
                # print(f"Detected changepoint at index: {curr_cp}")
                segment_data = data[curr_cp:seg_len, :]
                segment_data = segment_data.to(dtype=torch.float32)
                segment_data_3d = reshape_to_3d(segment_data, window_size)
                # segment_data_3d = normalize_tensor(segment_data_3d)
                data_seg = data[changepoints[-2]:curr_cp, :]
                data_seg_3d = reshape_to_3d(data_seg, window_size)
                # data_seg_3d = normalize_tensor(data_seg_3d)
                candidate_score = candidate_score[:changepoints[-2]]

                if len(weights_temp) == len(model_name):
                    weights_temp = [0] * len(model_name)
                    for i in range(len(model_name)):
                        weights_temp.append(0)
                    weights_temp[len(changepoints)-1] = 1
                    print('weights_temp:',weights_temp)
                    data_start = data[:curr_cp,:]
                    data_start_3d = reshape_to_3d(data_start,window_size)
                    results_zero = train_encoder_decoder(model_name,data_start, epochs,learning_rate,latent_space_size,hidden_dim,
                                                         batch_size,window_size,n_components)

                    for i, model in enumerate(model_name):
                        encoder = results_zero[model]['encoder']
                        decoder = results_zero[model]['decoder']

                        feature_experts[i].append(encoder)
                        recon_experts[i].append((encoder, decoder))

                    score_data_seg_unchanged, recon_losses_unchanged = compute_scores_for_3d_input(
                        data_start_3d, recon_experts, feature_experts, feature_means, feature_covariances,
                        train_data_means, weights_temp, lambda1, lambda2, lambda3)
                    print("score_data_seg_unchanged:",score_data_seg_unchanged)
                    print("recon_losses_unchanged:",recon_losses_unchanged)
                    candidate_score = np.append(candidate_score, score_data_seg_unchanged)

                if changepoint_type[curr_cp] == "weight":
                    weights_temp_unchanged = weights_temp.copy()
                    weights_temp_unchanged = np.append(weights_temp_unchanged,np.zeros(len(model_name)))
                    weights_temp = changed_weights
                    score_data_seg, recon_losses = compute_scores_for_3d_input(
                        segment_data_3d, recon_experts, feature_experts,
                        feature_means, feature_covariances, train_data_means,
                        weights_temp, lambda1, lambda2, lambda3)
                    for i in range(curr_cp, seg_len):
                        if score_data_seg[i - curr_cp] > tau:
                            y_pred[i] = 1

                    score_data_seg_unchanged, recon_losses_unchanged= compute_scores_for_3d_input(
                        data_seg_3d, recon_experts, feature_experts,
                        feature_means, feature_covariances, train_data_means,
                        weights_temp_unchanged, lambda1, lambda2, lambda3)
                    for i in range(changepoints[-2], curr_cp):
                        if score_data_seg_unchanged[i - changepoints[-2]] > tau:
                            y_pred[i] = 1

                    reconstruction_weighted = weighted_reconstruction(data_seg_3d, recon_experts, weights_temp)
                    candidate_score = np.append(candidate_score,score_data_seg_unchanged)
                    candidate_score = np.append(candidate_score,score_data_seg)

                    np.savez(os.path.join(f'.\models\segmentation_results_{changepoints[-2]}_{curr_cp}_{model_name}.npz'),
                             changepoints=changepoints, y_pred=y_pred, score_data_seg=score_data_seg,
                             recon_losses=recon_losses, reconstruction_data=reconstruction_weighted)

                if changepoint_type[curr_cp] == "condition":
                    weights_temp_unchanged = weights_temp.copy()
                    weights_temp_unchanged = np.append(weights_temp_unchanged,np.zeros(len(model_name)))
                    weights_temp = [0] * len(weights_temp)
                    for i in range(len(model_name)):
                        weights_temp.append(0)
                    count_condition = 1
                    for cp in changepoints:
                        if changepoint_type[cp] == 'condition':
                            count_condition += 1
                    weights_temp[count_condition] = 1
                    print("weights_temp_unchanged:", weights_temp_unchanged)
                    print("weights_temp:",weights_temp)

                    results = train_encoder_decoder(model_name, segment_data, epochs, learning_rate,
                                                    latent_space_size, hidden_dim, batch_size, window_size,n_components)

                    for i, model in enumerate(model_name):
                        encoder = results[model]['encoder']
                        decoder = results[model]['decoder']

                        feature_experts[i].append(encoder)
                        recon_experts[i].append((encoder, decoder))

                    score_data_seg, recon_losses = compute_scores_for_3d_input(segment_data_3d, recon_experts, feature_experts, feature_means,
                                                                               feature_covariances, train_data_means, weights_temp,
                                                                               lambda1, lambda2, lambda3)
                    y_pred_temp_index = [0] * (seg_len - curr_cp)
                    for i in range(curr_cp, seg_len):
                        if score_data_seg[i - curr_cp] > tau:
                            y_pred[i] = 1
                            y_pred_temp_index[i-curr_cp] = 1

                    score_data_seg_unchanged, recon_losses_unchanged= compute_scores_for_3d_input(
                        data_seg_3d, recon_experts, feature_experts, feature_means, feature_covariances,
                        train_data_means, weights_temp_unchanged, lambda1, lambda2, lambda3)
                    y_pred_temp_index_unchanged = [0] * (curr_cp - changepoints[-2])
                    for i in range(changepoints[-2], curr_cp):
                        if score_data_seg_unchanged[i - changepoints[-2]] > tau:
                            y_pred[i] = 1
                            y_pred_temp_index_unchanged[i-changepoints[-2]] = 1

                    for i, model in enumerate(model_name):
                        encoder = results[model]['encoder']
                        decoder = results[model]['decoder']
                        feature_extractor = encoder
                        reconstruction_extractor = (encoder,decoder)
                        feature_extractor_new, reconstruction_extractor_new, filtered_data = update_models_with_online_gradient_descent(feature_extractor,reconstruction_extractor,
                                                                                                                                    y_pred_temp_index,segment_data_3d,epochs,learning_rate,batch_size,window_size)
                        recon_experts[i].pop()
                        recon_experts[i].append(reconstruction_extractor_new)
                        feature_experts[i].pop()
                        feature_experts[i].append(feature_extractor_new)

                    reonctruction_new = []
                    with torch.no_grad():
                        data_seg_3d = data_seg_3d.to(device)
                        data_seg_3d = data_seg_3d.to(dtype=torch.float32)
                        reconstruction_output_unchanged = weighted_reconstruction(data_seg_3d, recon_experts, weights_temp_unchanged)
                        reonctruction_new.append(reconstruction_output_unchanged)
                    reconstructuion_new_array = np.concatenate([reonctruction_new[0]], axis=0)
                    if changepoints[-2] == 0:
                        candidate_score = np.append(candidate_score, score_data_seg)
                    else:
                        candidate_score = np.append(candidate_score,score_data_seg_unchanged)
                        candidate_score = np.append(candidate_score,score_data_seg)

                    np.savez(os.path.join(f'.\models\segmentation_results_{changepoints[-2]}_{curr_cp}_{model_name}.npz'),
                             changepoints=changepoints, y_pred=y_pred, score_data_seg = score_data_seg,
                             recon_losses = recon_losses, reconstruction_data = reconstructuion_new_array)

                seg_len += 1
            else:
                if seg_len >= length - 1:
                    data_seg = data[changepoints[-1]:length, :]
                    data_seg_3d = reshape_to_3d(data_seg, window_size)
                    # data_seg_3d = normalize_tensor(data_seg_3d)
                    candidate_score = candidate_score[:changepoints[-1]]

                    score_data_seg, recon_losses = compute_scores_for_3d_input(data_seg_3d, recon_experts, feature_experts,
                                                                 feature_means, feature_covariances, train_data_means,
                                                                 weights_temp, lambda1, lambda2, lambda3)
                    for i in range(changepoints[-1], length):
                        if score_data_seg[i - changepoints[-1]] > tau:
                            y_pred[i] = 1

                    reconstruction_weighted = weighted_reconstruction(data_seg_3d, recon_experts, weights_temp)
                    candidate_score = np.append(candidate_score, score_data_seg)
                    np.savez(os.path.join(f'.\models\segmentation_results_{changepoints[-1]}_{data.shape[0]}_{model_name}.npz'),
                             changepoints=changepoints, y_pred = y_pred, score_data_seg = score_data_seg,
                             recon_losses = recon_losses, reconstruction_data = reconstruction_weighted)
                for anomaly in anomaly_candidate:
                    if max(seg_cp) > anomaly and anomaly not in seg_cp:
                        y_pred[anomaly] = 1
                        anomaly_candidate.remove(anomaly)
                seg_len += 1
        else:
            for anomaly in anomaly_candidate:
                y_pred[anomaly] = 1
                anomaly_candidate.remove(anomaly)
            seg_len += 1

        if seg_len % data.shape[0] == 0:
            np.savez(os.path.join(f'.\models\segmentation_results_{seg_len}_{model_name}.npz'),
                     changepoints=changepoints, y_pred=y_pred, candidate_score = candidate_score,
                     candidate_recon_loss = candidate_recon_loss, candidate_likelihood_value = candidate_likelihood_value,
                     candidate_cos_sim = candidate_cos_sim, changepoint_type = changepoint_type, recon_experts = recon_experts,
                     feature_experts = feature_experts, weights = weights_temp)

        print("changepoints:",changepoints)
        for changepoint_cp in changepoints:
            print("changepoints_type:",changepoint_type[changepoint_cp])
    changepoints.append(data.shape[0])
    return changepoints, y_pred


def process_models_reconstruction(model_name, data_test, top_k = 3):
    """
    Process reconstruction data from multiple models and calculate weighted candidate_score_new

    Args:
        model_name: list, list of model names
        data_test: array, test data used to determine segmentation length

    Returns:
        recon_model: list of arrays, reconstruction data for each model
        candidate_score_new: array, final weighted candidate_score
    """
    seg_len = data_test.shape[0]
    recon_model = [[] for _ in range(len(model_name))]
    all_scores = [[] for _ in range(len(model_name))]
    all_recon_losses = [[] for _ in range(len(model_name))]

    for i, model in enumerate(model_name):
        try:
            model = [model]
            file_path = os.path.join(f'./models/segmentation_results_{seg_len}_{model}.npz')
            data = np.load(file_path, allow_pickle=True)

            if 'reconstruction_data' in data:
                recon_model[i] = data['reconstruction_data']
            elif 'recon_experts' in data:
                recon_model[i] = data['recon_experts']
            else:
                raise KeyError("No valid reconstruction data found in the file")

            if 'candidate_score' in data:
                all_scores[i] = data['candidate_score']
            if 'candidate_recon_loss' in data:
                all_recon_losses[i] = data['candidate_recon_loss']

        except Exception as e:
            print(f"Failed to load file {file_path} for model {model}: {e}")
            continue

    candidate_score_new = np.zeros_like(all_scores[0]) if len(all_scores[0]) > 0 else np.array([])

    top_k_indices = [[] for _ in range(len(all_scores[0]))]
    for i in range(len(all_scores[0])):
        column_values_with_indices = [
            (all_scores[j][i], j) for j in range(len(all_scores))
        ]

        sorted_column = sorted(
            column_values_with_indices,
            key=lambda x: x[0],
            reverse=False
        )[:top_k]
        top_k_indices[i] = [j for (value, j) in sorted_column]


    for i in range(len(all_scores[0])):
        for j in range(len(all_scores)):
            if j in top_k_indices[i]:
                candidate_temp = all_scores[j][i]
                candidate_score_new[i] += 1 / candidate_temp

    candidate_score_new = 1 / candidate_score_new
    return candidate_score_new