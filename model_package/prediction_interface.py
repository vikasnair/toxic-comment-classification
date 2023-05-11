
import pickle
from pathlib import Path

import pandas as pd

PACKAGE_PATH = Path(__file__).parent


class SklearnModel:
    def __init__(self):
        """This is where the serialized objects needed should
        be loaded as class attributes."""

        with open(PACKAGE_PATH / "model.pkl", "rb") as model_file:
            self.model = pickle.load(model_file)

    def predict_proba(self, input_data_df: pd.DataFrame):
        """Makes predictions with the model. Returns the class probabilities."""
        text_column = input_data_df.columns[0]
        return self.model.predict_proba(input_data_df[text_column])


def load_model():
    """Function that returns the wrapped model object."""
    return SklearnModel()
