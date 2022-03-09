# fantastic-monitor
Package for evaluating classification models

Given a path to `input_data.json` and `output_data.json`,
running `>> python main.py input_path output_path` will:
- read the inputs supplied in input_data.json
- obtain the classification metric supplied in `input_data.json`
- optionally, provide a confidence interval for the relevant metric based on a bootstrapped resampling
- write these values to `output_data.json`

The assumed form for `input_data.json` is a JSON with the following fields:
- `metric`: of type "str" that must be one of 'Accuracy', 'Sensitivity', 'Specificity', 'AUC', and 'SensitivitySpecificityEquivalencePoint'.
- `model_outputs`: list of list of float, with dimension (N,2). N examples with 2 outputs each (first entry = probability of class 0, second entry = probability of class 1).
- `gt_labels`: list of int, with dimension (N,). Ground truth class labels of the N measurements.
- `threshold`: float. Threshold for positive clasification for metrics with operating points (eg 'Accuracy').
- `ci`: bool. If True, include confidence interval in output.
- `num_bootstraps`: int. Number of iterations of bootstrapping to compute confidence interval.
- `alpha`: float. alpha parameter for confidence level of the interval.

`output_data.json` then includes:
- `value`: float. The value for `metric` based on the supplied inputs, truth labels, and threshold for positive classification.
- `lower_bound`: float, optional. The lower bound of the confidence interval for `value` if `ci` is True.
- `upper_bound`: float, optional. The upper bound of the confidence interval for `value` if `ci` is True.