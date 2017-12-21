from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def make_forest_cover_data():
    forest_cover = fetch_covtype()
    X, y = forest_cover.data, forest_cover.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=0)

    cols = ['Cover_Type', 'Elevation', 'Aspect', 'Slope',
            'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
            'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
            'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
            'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
            'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
            'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
            'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
            'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
            'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
            'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
            'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

    train_df = pd.DataFrame(np.hstack([y_train.reshape(-1, 1), X_train]),
                            columns=cols)
    test_df = pd.DataFrame(np.hstack([y_test.reshape(-1, 1), X_test]),
                           columns=cols)

    return train_df, test_df
