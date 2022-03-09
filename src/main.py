import sys
import os
from io_utils import parse_input_json, create_output_json
from metrics import get_metric

if __name__ == "__main__":
    cl_inputs = sys.argv
    if len(cl_inputs) != 3:
        raise ValueError("Incorrect number of command line arguments supplied.")
    
    input_json_path = cl_inputs[1]
    output_json_path = cl_inputs[2]
    if os.path.isfile(input_json_path):
        input_json = parse_input_json(input_json_path)
    else:
        raise ValueError("Input JSON not found.")
    
    value, lower_bound, upper_bound = get_metric(input_json)
    create_output_json(output_json_path, value, lower_bound, upper_bound)
    