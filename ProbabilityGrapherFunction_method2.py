import pandas as pd
import matplotlib.pyplot as plt
import csv
import datetime
from dateutil.parser import parse
import numpy as np
import pickle
from scipy.stats import multivariate_normal



def load_gmm_model():
    gmm_file_path = "/Users/apple/Desktop/gmm_model_v1.pkl"
    # Load the GMM model from the pickle file
    with open(gmm_file_path, 'rb') as file:
        gmm_model = pickle.load(file)

    return gmm_model


gmm_model = load_gmm_model()

# Define global variables
ThpUlMin = float('inf')
ThpUlMax = float('-inf')
ThpDlMin = float('inf')
ThpDlMax = float('-inf')


def PC(enb_id, sector_id, meas_time):
    global ThpUlMin, ThpUlMax, ThpDlMin, ThpDlMax

    # Define time window for measurements
    time_window = datetime.timedelta(minutes=14)

    # Convert meas_time to datetime object
    meas_time = parse(meas_time).time()  # Extract only the time component

    # Define file name and GMM features
    filename = '/Users/apple/Desktop/Mal_thrput.csv'
    gmm_features = ['DRB.IPThpDl.QCI7',
                    'DRB.IPThpUl.QCI7',
                    'DRB.UEActiveUl.QCI7',
                    'DRB.UEActiveDl.QCI7',
                    'DRB.PdcpSduBitrateUl.QCI7',
                    'DRB.PdcpSduBitrateDl.QCI7',
                    'DRB.PdcpSduDelayDl.QCI7',
                    'DRB.PdcpSduDropRateDl.QCI7',
                    'DRB.PdcpSduAirLossRateDl.QCI7',
                    'DRB.PdcpSduLossRateUl.QCI7']

    # Create an empty list to store matching matrices
    matrices = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Extract the time component from csv_meas_time
            csv_meas_time = parse(row['meas_time']).time()

            # Calculate the time difference
            time_diff = datetime.datetime.combine(datetime.date.min, meas_time) - datetime.datetime.combine(
                datetime.date.min, csv_meas_time)

            # Check if the enb_id, sector_id, and time fall within the specified window
            if int(row['enb_id']) == enb_id and int(row['sector_id']) == sector_id and abs(time_diff) <= time_window:
                try:
                    # Extract the relevant features for the GMM
                    features = [float(row[feature]) for feature in gmm_features]

                    # Append the features as a row matrix to the list
                    matrices.append(np.array(features).reshape(1, -1))

                    # Update ThpUlMin and ThpUlMax and ThpDlMin and ThpDlMax

                    ThpUlMin = min(ThpUlMin, features[1])
                    ThpUlMax = max(ThpUlMax, features[1])
                    ThpDlMin = min(ThpDlMin, features[0])
                    ThpDlMax = max(ThpDlMax, features[0])

                except ValueError:
                    print(f"Error converting features for row: {row}")

    # Check if any matching matrices were found
    if len(matrices) == 0:
        print("No matching data found.")
        return None

    # Calculate the average matrix
    global avg_matrix
    avg_matrix = np.mean(matrices, axis=0)

    # Calculate the predicted probabilities for each matrix in matrices
    probs_list = [gmm_model.predict_proba(matrix) for matrix in matrices]

    # Calculate the average predicted probabilities
    avg_probs = np.mean(probs_list, axis=0)

    return avg_probs


def create_data_point(UEActiveDl, PdcpSduBitrateDl, UEActiveUl, PdcpSduBitrateUl, value):
    return np.array([UEActiveDl, PdcpSduBitrateDl, UEActiveUl, PdcpSduBitrateUl, value])



def calculate_probabilities(enb_id, sector_id, meas_time, UEActiveDl, PdcpSduBitrateDl, UEActiveUl, PdcpSduBitrateUl,
                            gmm_model, indicesPYX, indicesPX, QCI, k):
    # Define time window for measurements
    time_window = datetime.timedelta(minutes=14)

    # Convert meas_time to datetime object
    meas_time = parse(meas_time).time()  # Extract only the time component

    # Define file name and GMM features
    filename = '/Users/apple/Desktop/Mal_thrput.csv'
    gmm_features = ['DRB.IPThpDl.QCI7',
                    'DRB.IPThpUl.QCI7',
                    'DRB.UEActiveUl.QCI7',
                    'DRB.UEActiveDl.QCI7',
                    'DRB.PdcpSduBitrateUl.QCI7',
                    'DRB.PdcpSduBitrateDl.QCI7',
                    'DRB.PdcpSduDelayDl.QCI7',
                    'DRB.PdcpSduDropRateDl.QCI7',
                    'DRB.PdcpSduAirLossRateDl.QCI7',
                    'DRB.PdcpSduLossRateUl.QCI7']

    # Create an empty list to store matching matrices
    matrices = []

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            # Extract the time component from csv_meas_time
            csv_meas_time = parse(row['meas_time']).time()

            # Calculate the time difference
            time_diff = datetime.datetime.combine(datetime.date.min, meas_time) - datetime.datetime.combine(
                datetime.date.min, csv_meas_time)

            # Check if the enb_id, sector_id, and time fall within the specified window
            if int(row['enb_id']) == enb_id and int(row['sector_id']) == sector_id and abs(time_diff) <= time_window:
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

    # Calculate the predicted probabilities for each matrix in matrices
    probs_list = [gmm_model.predict_proba(matrix) for matrix in matrices]

    # Calculate the average predicted probabilities
    avg_probs = np.mean(probs_list, axis=0)

    # Update ThpUlMin, ThpUlMax, ThpDlMin, ThpDlMax
    ThpUlMin = min(features[1] for features in matrices)
    ThpUlMax = max(features[1] for features in matrices)
    ThpDlMin = min(features[0] for features in matrices)
    ThpDlMax = max(features[0] for features in matrices)

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
    max_valueUl = data[attribute_list[1]].max()
    max_valueDl = data[attribute_list[0]].max()

    # Create the ThpUl list with values divided into 10^k parts
    global ThpUl
    if max_valueUl >= ThpUlMax * 3:
        ThpUl = np.linspace(ThpUlMin / 2, ThpUlMax * 3, 10 ** k)
    elif max_valueUl < ThpUlMax * 3:
        ThpUl = np.linspace(ThpUlMin / 2, max_valueUl, 10 ** k)

    # Create the ThpDl list with values divided into 10^k parts
    global ThpDl
    if max_valueDl >= ThpDlMax * 3:
        ThpDl = np.linspace(ThpDlMin / 2, ThpDlMax * 3, 10 ** k)
    elif max_valueDl < ThpDlMax * 3:
        ThpUl = np.linspace(ThpUlMin / 2, max_valueDl, 10 ** k)

    output = np.zeros((5, 10 ** k))
    for i in range(5):
        indices = indicesPYX if indicesPYX[4] == i else indicesPX
        mean = gmm_model.means_[i][indices]
        cov = gmm_model.covariances_[i][indices][:, indices]
        cov_psd = regularize_covariance_matrix(cov)

        for j, value in enumerate(ThpUl):
            data_point = create_data_point(UEActiveDl, PdcpSduBitrateDl, UEActiveUl, PdcpSduBitrateUl, value)
            rv = multivariate_normal(mean, cov_psd, allow_singular=True)
            output[i, j] = rv.pdf(data_point)

    return output


def regularize_covariance_matrix(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def PX(var1, var2, var3, var4, model):
    indicesPX = [1, 2, 7, 8]
    means_subset = model.means_[:, indicesPX]
    covariances_subset = model.covariances_[:, indicesPX][:, :, indicesPX]

    variables = [var1, var2, var3, var4]

    probabilities = np.zeros((1, model.n_components))

    for i in range(model.n_components):
        cluster_mean = means_subset[i]
        cluster_covariance = regularize_covariance_matrix(covariances_subset[i])

        cluster_probability = multivariate_normal.pdf(variables, mean=cluster_mean, cov=cluster_covariance,
                                                      allow_singular=True)
        probabilities[:, i] = cluster_probability

    return probabilities.reshape(-1, 1)


def final_output(enb_id, sector_id, meas_time, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k, gmm_model):
    # Call PC function to get probabilities
    pc_probs = PC(enb_id, sector_id, meas_time)

    # Call PYX function to get probabilities
    indicesPYX = [1, 2, 7, 8, 6]  # Adjust these indices according to the order of your features
    indicesPX = [1, 2, 7, 8]  # Adjust these indices according to the order of your features
    pyx_probs = calculate_probabilities(enb_id, sector_id, meas_time, UEActiveDl, BitrateDl, UEActiveUl, BitrateUl,
                                        gmm_model, indicesPYX, indicesPX, 7, k)

    # Call PX function to get probabilities
    px_probs = PX(UEActiveDl, BitrateDl, UEActiveUl, BitrateUl, gmm_model)

    # Multiply PC and PYX matrices
    pc_pyx = np.matmul(pc_probs, pyx_probs)

    # Calculate the determinant of PC * PX matrix
    pc_px_det = np.linalg.det(np.matmul(pc_probs, px_probs))

    # Divide PC * PYX by the determinant of PC * PX
    final_output_cluster = pyx_probs / pc_px_det
    final_output = pc_pyx / pc_px_det

    # Flatten the final output matrix
    final_output_flat = final_output.flatten()

    return final_output_flat, final_output_cluster


# Example usage
enb_id = 60104
sector_id = 0
meas_time = '2023-03-07 13:15:08+08:00'
UEActiveUl = 10
UEActiveDl = 20
BitrateUl = 2000
BitrateDl = 4000
k = 3

final_output_list = final_output(enb_id, sector_id, meas_time, UEActiveUl, UEActiveDl, BitrateUl, BitrateDl, k, gmm_model)

# Assuming 'final_output_list', 'ThpUl' and 'ThpDl' are defined before this code.
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Assuming 'final_output_list', 'ThpUl', and 'ThpDl' are defined before this code.
yUl = final_output_list[0]
xDl, yDl = np.meshgrid(ThpDl, ThpUl)  # Create a grid of ThpDl and ThpUl values

ax.plot_surface(xDl, yDl, yUl)
ax.set_xlabel('ThpDl')
ax.set_ylabel('ThpUl')
ax.set_zlabel('Probability')
ax.set_title('Probability Distribution')

# Find the maximum probability point
max_prob_index = np.unravel_index(np.argmax(yUl), yUl.shape)
max_prob_point = ThpDl[max_prob_index[1]], ThpUl[max_prob_index[0]], yUl[max_prob_index]

ax.scatter(max_prob_point[0], max_prob_point[1], max_prob_point[2], c='r', marker='o')  # plot max probability point

print("Highest probability:", max_prob_point)

plt.show()


fig, axs = plt.subplots(5, figsize=(10, 20))  # Adjust the overall size

for i in range(5):
    yCUl = (final_output_list[2][i, :])
    xCUl = ThpUl
    max_prob_index_CUl = np.argmax(yCUl)
    max_prob_point_CUl = xCUl[max_prob_index_CUl], yCUl[max_prob_index_CUl]

    axs[i].plot(xCUl, yCUl)
    axs[i].plot(max_prob_point_CUl[0], max_prob_point_CUl[1], 'ro')  # plot max probability point
    axs[i].set_xlabel('ThpUl')
    axs[i].set_ylabel(f'Probability for cluster {i}')
    axs[0].set_title('Probability Distribution ThpUl')
    axs[i].grid(True)
    print("Highest probability for ThpCUl with cluster number", i, "is:", max_prob_point_CUl)

plt.subplots_adjust(hspace=0.5)  # Adjust the horizontal space

fig, axs = plt.subplots(5, figsize=(10, 20))  # Adjust the overall size

for i in range(5):
    yCDl = (final_output_list[3][i, :])
    xCDl = ThpDl
    max_prob_index_CDl = np.argmax(yCDl)
    max_prob_point_CDl = xCDl[max_prob_index_CDl], yCDl[max_prob_index_CDl]

    axs[i].plot(xCDl, yCDl)
    axs[i].plot(max_prob_point_CDl[0], max_prob_point_CDl[1], 'ro')  # plot max probability point
    axs[i].set_xlabel('ThpDl')
    axs[i].set_ylabel(f'Probability for cluster {i}')
    axs[0].set_title('Probability Distribution ThpDl')
    axs[i].grid(True)
    print("Highest probability for ThpCDl with cluster number", i, "is:", max_prob_point_CDl)

plt.subplots_adjust(hspace=0.5)  # Adjust the horizontal space
plt.tight_layout()
plt.show()
