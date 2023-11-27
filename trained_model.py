import pickle
import dataclasses
import numpy as np
from typing import Optional, Sequence, Tuple
from thresholding_model_class import ThresholdingModel

def pre_trained_model(input_gauge):
    with open('thresholding_model.pkl', 'rb') as model_file:

        loaded_tm = pickle.load(model_file)

    # Use the loaded model to make predictions
    predicted_result = loaded_tm.infer(input_gauge)

    return predicted_result



