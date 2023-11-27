# import pickle
# import numpy as np
# from typing import Optional, Sequence, Tuple
# from thresholding_model_class import ThresholdingModel, GroundTruthMeasurement

# def pre_trained_model(input_gauge):
#     with open('thresholding_model.pkl', 'rb') as model_file:

#         loaded_tm = pickle.load(model_file)

#     # Use the loaded model to make predictions
#     predicted_result = loaded_tm.infer(input_gauge)

#     return predicted_result



# # result=pre_trained_model(12)
# # print(result)



# trained_model.py

import pickle
from thresholding_model_class import ThresholdingModel

def pre_trained_model(input_gauge):
    with open('thresholding_model.pkl', 'rb') as model_file:
        loaded_tm = pickle.load(model_file)

    # Use the loaded model to make predictions
    predicted_result = loaded_tm.infer(input_gauge)

    return predicted_result

# This block ensures that the following code is only executed when the script is run directly.
if __name__ == "__main__":
    # Additional code for model evaluation or other tasks when the script is run directly
    pass
