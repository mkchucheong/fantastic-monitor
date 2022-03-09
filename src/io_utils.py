import json
import numpy as np

def parse_input_json(input_path):
    try:
        with open(input_path, 'r') as myfile:
            data=json.load(myfile)
        
        # data should contain the following fields
        # metric, model_outputs, gt_labels, ci
        # optionally, threshold
        # ci
        required_keys = ["metric", "model_outputs", "gt_labels", "ci"]
        for k in required_keys:
            if k not in data.keys():
                raise Exception
        
        # metric
        accepted_metrics = ["Accuracy", "AUC", "Sensitivity", "Specificity", 
        "SensitivitySpecificityEquivalencePoint"]
        if data["metric"] not in accepted_metrics:
            raise KeyError("Supplied metric not found")
        
        # model_outputs
        data['model_outputs'] = np.asarray(data['model_outputs'])

        # gt_labels
        data['gt_labels'] = np.asarray(data['gt_labels'])
        
        return data
    except:
        print("Input does not match specifications.")

def create_output_json(output_path, value, lower_bound, upper_bound):
    if lower_bound is None:
        out_dict = {"value": value}
    else:
        out_dict = {"value": value, "lower_bound": lower_bound, "upper_bound": upper_bound}
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4)