import pickle



def load_gmm_model():
    gmm_file_path = "/Users/apple/Desktop/gmm_model_v1.pkl"
    # Load the GMM model from the pickle file
    with open(gmm_file_path, 'rb') as file:
        gmm_model = pickle.load(file)

    return gmm_model


gmm_model = load_gmm_model()

import csv
import datetime
from dateutil.parser import parse
import numpy as np

def PC(enb_id, sector_id, meas_time):
    # Define time window for measurements
    time_window = datetime.timedelta(minutes=14)

    # Convert meas_time to datetime object
    meas_time = parse(meas_time)

    # Define file name and GMM features
    filename = '/Users/apple/Desktop/Mal_thrput.csv'
    gmm_features = ['DRB.IPThpDl.QCI7', 'DRB.UEActiveDl.QCI7', 'DRB.PdcpSduBitrateDl.QCI7',
                    'DRB.PdcpSduDelayDl.QCI7', 'DRB.PdcpSduDropRateDl.QCI7', 'DRB.PdcpSduAirLossRateDl.QCI7',
                    'DRB.IPThpUl.QCI7', 'DRB.UEActiveUl.QCI7', 'DRB.PdcpSduBitrateUl.QCI7',
                    'DRB.PdcpSduLossRateUl.QCI7']

    # Create an empty list to store matching matrices
    matrices = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Convert meas_time from the CSV to datetime object
            csv_meas_time = parse(row['meas_time'])

            # Check if the enb_id, sector_id, and time fall within the specified window
            if int(row['enb_id']) == enb_id and int(row['sector_id']) == sector_id and abs(meas_time - csv_meas_time) <= time_window:
                try:
                    # Extract the relevant features for the GMM
                    features = [float(row[feature]) for feature in gmm_features]

                    # Append the features as a row matrix to the list
                    matrices.append(np.array(features).reshape(1, -1))
                except ValueError:
                    print(f"Error converting features for row: {row}")

    # Check if any matching matrices were found
    if len(matrices) == 0:
        print("No matching data found.")
        return None

    # Calculate the average matrix
    avg_matrix = np.mean(matrices, axis=0)

    # Load the pre-existing GMM model
    gmm_model = load_gmm_model()  # Replace with your code to load the GMM model

    # Get the predicted probabilities
    probs = gmm_model.predict_proba(avg_matrix)

    return probs


# Example usage of the PC function
enb_id = 60104  # Replace with your desired enb_id
sector_id = 1  # Replace with your desired sector_id
meas_time = '2023-03-04 22:00:00+08:00'  # Replace with your desired meas_time

probs = PC(enb_id, sector_id, meas_time)
print(probs)



import pickle


def load_gmm_model():
    gmm_file_path = "/Users/apple/Desktop/gmm_model_v1.pkl"
    # Load the GMM model from the pickle file
    with open(gmm_file_path, 'rb') as file:
        gmm_model = pickle.load(file)

    return gmm_model


gmm_model = load_gmm_model()




import numpy as np
import pandas as pd

def PYX(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k):
    # Dictionary mapping QCI to attribute names
    attribute_names = {
        1: ["DRB.IPThpDl.QCI1", "DRB.IPThpUl.QCI1"],
        2: ["DRB.IPThpDl.QCI2", "DRB.IPThpUl.QCI2"],
        3: ["DRB.IPThpDl.QCI3", "DRB.IPThpUl.QCI3"],
        4: ["DRB.IPThpDl.QCI4", "DRB.IPThpUl.QCI4"],
        5: ["DRB.IPThpDl.QCI5", "DRB.IPThpUl.QCI5"],
        6: ["DRB.IPThpDl.QCI6", "DRB.IPThpUl.QCI6"],
        7: ["DRB.IPThpDl.QCI7", "DRB.IPThpUl.QCI7"],
        8: ["DRB.IPThpDl.QCI8", "DRB.IPThpUl.QCI8"],
        9: ["DRB.IPThpDl.QCI9", "DRB.IPThpUl.QCI9"]
    }

    # Read the CSV file
    data = pd.read_csv('/Users/apple/Desktop/Mal_thrput.csv')

    # Get the attribute names for the given QCI
    attribute_list = attribute_names[QCI]

    # Get the range of values for DRB.IPThpUl.QCI
    min_value = data[attribute_list[1]].min()
    max_value = data[attribute_list[1]].max()

    # Create the ThpUl list with values divided into 10^k parts
    ThpUl = np.linspace(min_value, max_value, 10 ** k)

    # Create the data matrix with 5 rows and 10^k columns
    data_matrix = np.zeros((5, 10 ** k))

    # Loop through each value in ThpUl
    for i, value in enumerate(ThpUl):
        # Assign the input values to the corresponding columns in data matrix
        data_matrix[0, i] = value  # DRB.IPThpUl.QCI7
        data_matrix[1, i] = UEActiveUl  # DRB.UEActiveUl.QCI7
        data_matrix[2, i] = UEActiveDl  # DRB.UEActiveDl.QCI7
        data_matrix[3, i] = BitrateUl  # DRB.PdcpSduBitrateUl.QCI7
        data_matrix[4, i] = BitrateDl  # DRB.PdcpSduBitrateDl.QCI7

    # Placeholder columns (set to 0)
    placeholder_columns = np.zeros((5, 10 ** k))

    # Concatenate the data matrix and placeholder columns
    input_data = np.concatenate((data_matrix, placeholder_columns), axis=0)

    # Get the GMM model predictions for the input data
    predictions = gmm_model.predict_proba(input_data.T)

    # Transpose the predictions and return as the result
    return predictions.T

# Example usage
QCI = 7
UEActiveUl = 100
UEActiveDl = 200
BitrateUl = 10
BitrateDl = 20
k = 3

result = PYX(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k)
print(result.shape)

result = PYX(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k)
print(result)



import pickle


import numpy as np

def load_gmm_model():
    gmm_file_path = "/Users/apple/Desktop/gmm_model_v1.pkl"
    # Load the GMM model from the pickle file
    with open(gmm_file_path, 'rb') as file:
        gmm_model = pickle.load(file)

    return gmm_model


gmm_model = load_gmm_model()


import numpy as np

def PX(UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, gmm_model):
    # Define a dictionary to map each QCI to the relevant attribute names
    attribute_names = {
        1: ["DRB.IPThpDl.QCI1", "DRB.IPThpUl.QCI1"],
        2: ["DRB.IPThpDl.QCI2", "DRB.IPThpUl.QCI2"],
        3: ["DRB.IPThpDl.QCI3","DRB.IPThpUl.QCI3"],
        4: ["DRB.IPThpDl.QCI4","DRB.IPThpUl.QCI4"],
        5: ["DRB.IPThpDl.QCI5","DRB.IPThpUl.QCI5"],
        6: ["DRB.IPThpDl.QCI6","DRB.IPThpUl.QCI6"],
        7: ["DRB.IPThpDl.QCI7", "DRB.IPThpUl.QCI7"],
        8: ["DRB.IPThpDl.QCI8","DRB.IPThpUl.QCI8"],
        9: ["DRB.IPThpDl.QCI9","DRB.IPThpUl.QCI9"]
    }

    # Select the attribute names for QCI 7 from the dictionary
    qci_attributes = attribute_names[7]

    # Assign input values to the corresponding feature variables
    DRB_IPThpUl_QCI7 = 0  # Placeholder value
    DRB_UEActiveUl_QCI7 = UEActiveUl
    DRB_UEActiveDl_QCI7 = UEActiveDl
    DRB_PdcpSduBitrateUl_QCI7 = BitrateUl
    DRB_PdcpSduBitrateDl_QCI7 = BitrateDl

    # Assign placeholder values (0) to other features
    DRB_IPThpDl_QCI7 = 0
    DRB_PdcpSduDelayDl_QCI7 = 0
    DRB_PdcpSduDropRateDl_QCI7 = 0
    DRB_PdcpSduAirLossRateDl_QCI7 = 0
    DRB_PdcpSduLossRateUl_QCI7 = 0

    # Create a data point with the attribute values
    data_point = np.array([
        DRB_IPThpDl_QCI7,
        DRB_IPThpUl_QCI7,
        DRB_UEActiveUl_QCI7,
        DRB_UEActiveDl_QCI7,
        DRB_PdcpSduBitrateUl_QCI7,
        DRB_PdcpSduBitrateDl_QCI7,
        DRB_PdcpSduDelayDl_QCI7,
        DRB_PdcpSduDropRateDl_QCI7,
        DRB_PdcpSduAirLossRateDl_QCI7,
        DRB_PdcpSduLossRateUl_QCI7
    ])

    # Get the probabilities of the data point belonging to each cluster
    probabilities_qci7 = gmm_model.predict_proba(data_point.reshape(1, -1))

    # Create a column matrix with the probabilities for QCI 7
    column_matrix = probabilities_qci7.reshape(-1, 1)

    return column_matrix

# Assume you have a pre-trained GMM model named 'gmm_model'
UEActiveUl = 50
UEActiveDl = 60
BitrateUl = 100
BitrateDl = 200

result = PX(UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, gmm_model)
print(result)

# Function PC
import csv
import datetime
import numpy as np
from dateutil.parser import parse

def PC(enb_id, sector_id, meas_time):
    # Define time window for measurements
    time_window = datetime.timedelta(minutes=14)

    # Convert meas_time to datetime object
    meas_time = parse(meas_time)

    # Define file name and GMM features
    filename = '/Users/apple/Desktop/Mal_thrput.csv'
    gmm_features = ['DRB.IPThpDl.QCI7', 'DRB.UEActiveDl.QCI7', 'DRB.PdcpSduBitrateDl.QCI7',
                    'DRB.PdcpSduDelayDl.QCI7', 'DRB.PdcpSduDropRateDl.QCI7', 'DRB.PdcpSduAirLossRateDl.QCI7',
                    'DRB.IPThpUl.QCI7', 'DRB.UEActiveUl.QCI7', 'DRB.PdcpSduBitrateUl.QCI7',
                    'DRB.PdcpSduLossRateUl.QCI7']

    # Create an empty list to store matching matrices
    matrices = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Convert meas_time from the CSV to datetime object
            csv_meas_time = parse(row['meas_time'])

            # Check if the enb_id, sector_id, and time fall within the specified window
            if int(row['enb_id']) == enb_id and int(row['sector_id']) == sector_id and abs(meas_time - csv_meas_time) <= time_window:
                try:
                    # Extract the relevant features for the GMM
                    features = [float(row[feature]) for feature in gmm_features]

                    # Append the features as a row matrix to the list
                    matrices.append(np.array(features).reshape(1, -1))
                except ValueError:
                    print(f"Error converting features for row: {row}")

    # Check if any matching matrices were found
    if len(matrices) == 0:
        print("No matching data found.")
        return None

    # Calculate the average matrix
    avg_matrix = np.mean(matrices, axis=0)

    # Load the pre-existing GMM model
    gmm_model = load_gmm_model()  # Replace with your code to load the GMM model

    # Get the predicted probabilities
    probs = gmm_model.predict_proba(avg_matrix)

    return probs


# Function PYX
import pandas as pd
import numpy as np

def PYX(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k):
    # Dictionary mapping QCI to attribute names
    attribute_names = {
        1: ["DRB.IPThpDl.QCI1", "DRB.IPThpUl.QCI1"],
        2: ["DRB.IPThpDl.QCI2", "DRB.IPThpUl.QCI2"],
        3: ["DRB.IPThpDl.QCI3", "DRB.IPThpUl.QCI3"],
        4: ["DRB.IPThpDl.QCI4", "DRB.IPThpUl.QCI4"],
        5: ["DRB.IPThpDl.QCI5", "DRB.IPThpUl.QCI5"],
        6: ["DRB.IPThpDl.QCI6", "DRB.IPThpUl.QCI6"],
        7: ["DRB.IPThpDl.QCI7", "DRB.IPThpUl.QCI7"],
        8: ["DRB.IPThpDl.QCI8", "DRB.IPThpUl.QCI8"],
        9: ["DRB.IPThpDl.QCI9", "DRB.IPThpUl.QCI9"]
    }

    # Read the CSV file
    data = pd.read_csv('/Users/apple/Desktop/Mal_thrput.csv')

    # Get the attribute names for the given QCI
    attribute_list = attribute_names[QCI]

    # Get the range of values for DRB.IPThpUl.QCI
    min_value = data[attribute_list[1]].min()
    max_value = data[attribute_list[1]].max()

    # Create the ThpUl list with values divided into 10^k parts
    ThpUl = np.linspace(min_value, max_value, 10 ** k)

    # Create the data matrix with 5 rows and 10^k columns
    data_matrix = np.zeros((5, 10 ** k))

    # Loop through each value in ThpUl
    for i, value in enumerate(ThpUl):
        # Assign the input values to the corresponding columns in data matrix
        data_matrix[0, i] = value  # DRB.IPThpUl.QCI7
        data_matrix[1, i] = UEActiveUl  # DRB.UEActiveUl.QCI7
        data_matrix[2, i] = UEActiveDl  # DRB.UEActiveDl.QCI7
        data_matrix[3, i] = BitrateUl  # DRB.PdcpSduBitrateUl.QCI7
        data_matrix[4, i] = BitrateDl  # DRB.PdcpSduBitrateDl.QCI7

    # Placeholder columns (set to 0)
    placeholder_columns = np.zeros((5, 10 ** k))

    # Concatenate the data matrix and placeholder columns
    input_data = np.concatenate((data_matrix, placeholder_columns), axis=0)

    # Get the GMM model predictions for the input data
    predictions = gmm_model.predict_proba(input_data.T)

    # Transpose the predictions and return as the result
    return predictions.T


# Function PX
import numpy as np

def PX(UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, gmm_model):
    # Define a dictionary to map each QCI to the relevant attribute names
    attribute_names = {
        1: ["DRB.IPThpDl.QCI1", "DRB.IPThpUl.QCI1"],
        2: ["DRB.IPThpDl.QCI2", "DRB.IPThpUl.QCI2"],
        3: ["DRB.IPThpDl.QCI3","DRB.IPThpUl.QCI3"],
        4: ["DRB.IPThpDl.QCI4","DRB.IPThpUl.QCI4"],
        5: ["DRB.IPThpDl.QCI5","DRB.IPThpUl.QCI5"],
        6: ["DRB.IPThpDl.QCI6","DRB.IPThpUl.QCI6"],
        7: ["DRB.IPThpDl.QCI7", "DRB.IPThpUl.QCI7"],
        8: ["DRB.IPThpDl.QCI8","DRB.IPThpUl.QCI8"],
        9: ["DRB.IPThpDl.QCI9","DRB.IPThpUl.QCI9"]
    }

    # Select the attribute names for QCI 7 from the dictionary
    qci_attributes = attribute_names[7]

    # Assign input values to the corresponding feature variables
    DRB_IPThpUl_QCI7 = 0  # Placeholder value
    DRB_UEActiveUl_QCI7 = UEActiveUl
    DRB_UEActiveDl_QCI7 = UEActiveDl
    DRB_PdcpSduBitrateUl_QCI7 = BitrateUl
    DRB_PdcpSduBitrateDl_QCI7 = BitrateDl

    # Assign placeholder values (0) to other features
    DRB_IPThpDl_QCI7 = 0
    DRB_PdcpSduDelayDl_QCI7 = 0
    DRB_PdcpSduDropRateDl_QCI7 = 0
    DRB_PdcpSduAirLossRateDl_QCI7 = 0
    DRB_PdcpSduLossRateUl_QCI7 = 0

    # Create a data point with the attribute values
    data_point = np.array([
        DRB_IPThpDl_QCI7,
        DRB_IPThpUl_QCI7,
        DRB_UEActiveUl_QCI7,
        DRB_UEActiveDl_QCI7,
        DRB_PdcpSduBitrateUl_QCI7,
        DRB_PdcpSduBitrateDl_QCI7,
        DRB_PdcpSduDelayDl_QCI7,
        DRB_PdcpSduDropRateDl_QCI7,
        DRB_PdcpSduAirLossRateDl_QCI7,
        DRB_PdcpSduLossRateUl_QCI7
    ])

    # Get the probabilities of the data point belonging to each cluster
    probabilities_qci7 = gmm_model.predict_proba(data_point.reshape(1, -1))

    # Create a column matrix with the probabilities for QCI 7
    column_matrix = probabilities_qci7.reshape(-1, 1)

    return column_matrix

import numpy as np

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def final_output_graph(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k, gmm_model,enb_id, sector_id, meas_time):
    # Call PC function
    pc_result = PC(enb_id, sector_id, meas_time)

    # Call PYX function
    pyx_result = PYX(QCI, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k)

    # Call PX function
    px_result = PX(UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, gmm_model)

    # Multiply PC and PYX matrices
    pc_pyx_product = np.multiply(pc_result, pyx_result)

    # Multiply PC and PX matrices
    pc_px_product = np.multiply(pc_result, px_result)

    # Calculate the determinant of PC * PX matrix
    determinant = np.linalg.det(pc_px_product)

    # Divide the PC * PYX matrix by the determinant
    final_output = np.divide(pc_pyx_product, determinant)

    # Reshape final output matrix into a 1D array
    final_output_1d = final_output.flatten()

    # Create histogram plot using seaborn
    sns.histplot(final_output_1d, kde=True)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Final Output Histogram')
    plt.show()

    return final_output
