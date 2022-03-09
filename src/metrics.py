import numpy as np
import pandas as pd
from scipy.integrate import trapz

def get_prediction_labels(predictions, threshold):
    pred_labels = predictions[:,1].reshape(-1) >= threshold
    return pred_labels.astype(int)

def get_roc_vectorized(predictions, truth, dt):
    thresholds = np.arange(0, 1, dt)
    nt = thresholds.shape[0]
    roc = np.zeros((3, nt))
    roc[0, :] = thresholds
    
    pred_labels = predictions[:,1].reshape((-1, 1)) >= thresholds * np.ones((predictions.shape[0], nt))
    # sensitivity
    obs_positives = np.sum(truth==1)
    labeled_pos = np.sum(pred_labels[truth==1, :], axis=0)
    # account for 0 denom
    #labeled_pos[obs_positives == 0] = 1
    #obs_positives[obs_positives == 0] = 1
    
    
    # specificity
    obs_negatives = np.sum(truth==0)
    labeled_neg = np.sum(pred_labels[truth==0, :]==0, axis=0)
    #labeled_neg[obs_negatives == 0] = 1
    #obs_negatives[obs_negatives == 0] = 1
    
    if obs_negatives == 0:
        roc[1,:] = np.zeros(nt)
    else:
        roc[1,:] = 1-(labeled_neg/obs_negatives)
    if obs_positives == 0:
        roc[2,:] = np.ones(nt)
    else:
        roc[2,:] = labeled_pos/obs_positives
    return roc
    

def lin_rootfind(coord1, coord2):
    x1,y1 = coord1
    x2,y2 = coord2

    m = (y2-y1)/(x2-x1)

    root = (y1 / m) + x1
    return root

def get_sensitivity(predictions, threshold, truth):
    labels = get_prediction_labels(predictions, threshold)
    # subset of labels for which the observed data is true
    obs_positives = labels[truth == 1]
    
    # number of total positives
    n_positives = len(obs_positives)
    
    if n_positives == 0:
        # no positives exist in the data
        return 1
    else:
        # number of correctly labeled positives / number of total positives
        tpr = np.sum(obs_positives) / n_positives
        return tpr

def get_specificity(predictions, threshold, truth):
    labels = get_prediction_labels(predictions, threshold)
    obs_negatives = labels[truth == 0]
    n_negatives = len(obs_negatives)
    if n_negatives == 0:
        return 1
    else:
        tnr = np.sum(obs_negatives==0) / n_negatives
        return tnr

def get_accuracy(predictions, threshold, truth):
    labels = get_prediction_labels(predictions, threshold)
    return np.sum(labels == truth) / len(truth)

def get_roc_auc(predictions, threshold, truth):
    roc = get_roc_vectorized(predictions, truth, dt=1E-2)
    
    # sensitivity
    y = roc[2,:]
    
    # 1 - specificity
    x = roc[1,:]
    auc = trapz(y=y[::-1], x=x[::-1])
    return auc

def get_sens_spec_equivalence(predictions, threshold, truth):
    roc = get_roc_vectorized(predictions, truth, dt=1E-2)
    t = roc[0,:]
    spec = 1 - roc[1,:]
    sens = roc[2,:]

    delta = spec-sens
    pos_subset = delta >=0
    neg_subset = delta <= 0
    if len(pos_subset)== 0:
        greatest_neg_idx = np.argmax(delta[neg_subset])
        nearest_t = t[neg_subset][greatest_neg_idx]
    elif len(neg_subset) == 0:
        least_pos_idx = np.argmin(delta[pos_subset])
        nearest_t = t[pos_subset][least_pos_idx]
    else:
        greatest_neg_idx = np.argmax(delta[neg_subset])
        least_pos_idx = np.argmin(delta[pos_subset])
        neg_t = t[neg_subset][greatest_neg_idx]
        neg_val = delta[neg_subset][greatest_neg_idx]
        pos_t = t[pos_subset][least_pos_idx]
        pos_val = delta[pos_subset][least_pos_idx]
        
        if neg_t == pos_t:
            nearest_t = pos_t
        else:
            nearest_t = lin_rootfind((neg_t, neg_val), (pos_t, pos_val))

    return nearest_t     

def get_ci(fun, predictions, threshold, truth, n_trials, alpha):
    n = len(predictions)
    indices = np.arange(0, n)
    
    bootstrapped_results = np.zeros((n_trials, 1))
    for i in range(n_trials):
        resampled_indices = np.random.choice(indices, n)
        resampled_predictions = predictions[resampled_indices, :]
        resampled_truth = truth[resampled_indices]
        
        value = fun(resampled_predictions, threshold, resampled_truth)
        bootstrapped_results[i] = value
    
    # get alpha'th percentile
    lower_bound = np.percentile(bootstrapped_results, alpha)
    
    # get 1-alpha'th percentile
    upper_bound = np.percentile(bootstrapped_results, 100-alpha)
    
    return lower_bound, upper_bound 

def get_metric(data):
    metric = data["metric"]
    threshold = data["threshold"]
    predictions = data["model_outputs"]
    truth = data["gt_labels"]
    ci_flag = data["ci"]
    num_trials = data["num_bootstraps"]
    alpha = data["alpha"]*100
    
    relevant_function = {}
    relevant_function["Accuracy"] = get_accuracy
    relevant_function["Sensitivity"] = get_sensitivity
    relevant_function["Specificity"] = get_specificity
    relevant_function["AUC"] = get_roc_auc
    relevant_function["SensitivitySpecificityEquivalencePoint"] = get_sens_spec_equivalence
    
    specified_function = relevant_function[metric]
    value = specified_function(predictions, threshold, truth)
    if ci_flag:
        lower_bound, upper_bound = get_ci(specified_function, predictions, threshold, truth, num_trials, alpha)
        return value, lower_bound, upper_bound
    else:
        return value, None, None