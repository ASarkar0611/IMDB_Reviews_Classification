import pandas as pd
import pickle


class ClscommonUtils:
    @staticmethod
    def load_data(path):
        return pd.read_csv(path)

