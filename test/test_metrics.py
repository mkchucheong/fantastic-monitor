import pytest
import numpy as np
import sys
sys.path.append('../src')
from metrics import get_accuracy, get_sensitivity, get_sens_spec_equivalence, get_roc_auc, get_specificity, get_ci

def test_accuracy():
    n = 10
    predictions = np.zeros((n,2))
    threshold = 0.5
    labels = np.ones(n)
    
    output_accuracy = 0
    assert get_accuracy(predictions, threshold, labels) == output_accuracy
    
    predictions[:,1] = 1
    output_accuracy = 1
    assert get_accuracy(predictions, threshold, labels) == output_accuracy
    
    predictions[:,1] = np.linspace(0, 1, n)
    output_accuracy = 0.5
    assert get_accuracy(predictions, threshold, labels) == output_accuracy

def test_sensitivity():
    # sensitivity = True positives / Positives
    n = 10
    predictions = np.zeros((n,2))
    threshold = 0.5
    
    # 0 positives, 0 true positives- sensitivity = 1
    labels = np.zeros(n)
    output_sens = 1
    assert get_sensitivity(predictions, threshold, labels) == output_sens
    
    # 0 positives, all predicted positive (still 0 true positives)- sens = 1
    labels = np.zeros(n)
    predictions[:,1] = np.ones(n)
    output_sens = 1
    assert get_sensitivity(predictions, threshold, labels) == output_sens
    
    # N positives, 0 true positives- sens = 0
    labels = np.ones(n)
    predictions[:,1] = np.zeros(n)
    output_sens = 0
    assert get_sensitivity(predictions, threshold, labels) == output_sens
    
    # N positives, N true positives- sens = 1
    predictions[:,1] = np.ones(n)
    output_sens = 1
    assert get_sensitivity(predictions, threshold, labels) == output_sens
    
    # N positives, N/2 true positives- sens = 0.5
    output_sens = 0.5
    predictions[:,1] = np.linspace(0, 1, n)
    assert get_sensitivity(predictions, threshold, labels) == output_sens
    
    
def test_specificity():
    # specificity = True negatives / Negatives
    n = 10
    predictions = np.ones((n,2))
    threshold = 0.5
    
    # 0 negatives, 0 true negatives- specificity = 1
    labels = np.ones(n)
    output_spec = 1
    assert get_specificity(predictions, threshold, labels) == output_spec
    
    # 0 negatives, all predicted negative (0 true negatives)- specificity = 1
    labels = np.ones(n)
    output_spec = 1
    assert get_specificity(predictions, threshold, labels) == output_spec
    
    # N negatives, 0 true negatives- specificity = 0
    labels = np.zeros(n)
    output_spec=0
    assert get_specificity(predictions, threshold, labels) == output_spec
    
    # N negatives, N true negatives - specificity   = 1
    labels = np.zeros(n)
    predictions[:,1] = np.zeros(n)
    output_spec = 1
    assert get_specificity(predictions, threshold, labels) == output_spec
    
    # N negatives, N/2 true negatives- specificity = 0.5
    output_spec = 0.5
    predictions[:,1] = np.linspace(0, 1, n)
    assert get_specificity(predictions, threshold, labels) == output_spec
    
def test_auc():
    predictions= np.zeros((10, 2))
    predictions[:,1] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels =           np.array([0,   0,   1,   0,   1,   1,   0,   1,   1,   1])
    # 6 true, 4 false
    # threshold = 0: 0 labeled false, : sensitivity = 6/6=1, spec = 0/4 =0
    # threshold = 0.1, 10 labeled true: sensitivity = 6/6, spec=1/4
    # threshold = 0.2, 9 labeled true: sensitivity = 6/6, spec = 2/4
    # threshold = 0.3, 8 labeled true: sensitivity = 5/6, spec = 2/4
    # threshold = 0.4, 7 labeled true: sensitivity = 5/6, spec = 3/4
    # threshold = 0.5, 6 labeled true: sensitivity = 4/6, spec = 3/4
    # threshold = 0.6, 5 labeled true: sensitivity = 3/6, spec = 3/4
    # threshold = 0.7, 4 labeled true: sensitivity = 3/6, spec = 4/4
    # threshold = 0.8, 3 labeled true: sensitivity = 2/6, spec = 4/4
    # threshold = 0.9, 2 labeled true: sensitivity = 1/6, spec = 4/4
    # threshold = 1.0, 1 labeled true: sensitivity = 0/6, spec = 4/4
    known_auc = 0.8333333333333334
    roc_auc = get_roc_auc(predictions, 0.5, labels)
    tol = 1E-8
    assert roc_auc - known_auc < tol

    predictions = np.zeros((10, 2))
    labels = np.ones(10)
    roc_auc = get_roc_auc(predictions, 0.5, labels)
    assert roc_auc == 0

def test_sens_spec_equivalence():
    predictions= np.zeros((10, 2))
    predictions[:,1] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels =           np.array([0,   0,   1,   0,   1,   1,   0,   1,   1,   1])
    # 6 true, 4 false
    # threshold = 0: 0 labeled false, : sensitivity = 6/6=1, spec = 0/4 =0
    # threshold = 0.1, 10 labeled true: sensitivity = 6/6, spec=1/4
    # threshold = 0.2, 9 labeled true: sensitivity = 6/6, spec = 2/4
    # threshold = 0.3, 8 labeled true: sensitivity = 5/6, spec = 2/4
    # threshold = 0.4, 7 labeled true: sensitivity = 5/6, spec = 3/4
    # threshold = 0.5, 6 labeled true: sensitivity = 4/6, spec = 3/4
    # threshold = 0.6, 5 labeled true: sensitivity = 3/6, spec = 3/4
    # threshold = 0.7, 4 labeled true: sensitivity = 3/6, spec = 4/4
    # threshold = 0.8, 3 labeled true: sensitivity = 2/6, spec = 4/4
    # threshold = 0.9, 2 labeled true: sensitivity = 1/6, spec = 4/4
    # threshold = 1.0, 1 labeled true: sensitivity = 0/6, spec = 4/4
    known_midpoint = 0.45
    sens_spec_midpoint = get_sens_spec_equivalence(predictions, 0.5, labels)
    tol = 1E-8
    assert sens_spec_midpoint - known_midpoint < tol

    predictions = np.zeros((10, 2))
    labels = np.ones(10)
    sens_spec_midpoint = get_sens_spec_equivalence(predictions, 0.5, labels)
    assert sens_spec_midpoint == 0