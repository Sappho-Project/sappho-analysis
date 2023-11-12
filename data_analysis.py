import numpy as np
from scipy import signal
import os
from scipy.spatial.distance import euclidean
from scipy.fft import fft
import pywt
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp, linregress, pearsonr
from sklearn.metrics import r2_score
import math

def import_sanitised_data(file_path="sanitised_data.txt"):
    data = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        # Remove leading/trailing whitespace and brackets
        line = line.strip().strip("[]")
        elements = line.split(", ")

        # Convert elements to floats and append them to the data list
        data.append([float(element) for element in elements])

    return data


def normalize_data(data):
    normalized_data = []

    for arr in data:
        min_value = min(arr)
        max_value = max(arr)
        normalized_array = [(x - min_value) / (max_value - min_value) for x in arr]
        normalized_data.append(normalized_array)

    return normalized_data


def write_results_to_file(results, filename):
    # Create "results" directory if it doesn't exist
    if not os.path.exists("results"):
        os.mkdir("results")

    file_path = os.path.join("results", filename)

    with open(file_path, "w") as file:
        # Write the results to the file, clearing its content if the file already exists
        for result in results:
            file.write(str(result) + '\n')


def coherence(theoretical_data, experimental_data):
    coherences = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data:
            f, Pxy = signal.csd(theoretical_array, experimental_array, fs=1, nperseg=64)
            f, Pxx = signal.csd(theoretical_array, theoretical_array, fs=1, nperseg=64)
            f, Pyy = signal.csd(experimental_array, experimental_array, fs=1, nperseg=64)

            coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy)
            mean_coherence = np.mean(coherence)

            coherences.append(mean_coherence)

    write_results_to_file(coherences, "coherence_results.txt")


def cross_correlation(theoretical_data, experimental_data):
    cross_correlations = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data:
            result = signal.correlate(theoretical_array, experimental_array, mode="full")

            cross_correlations.append(result.tolist())

    write_results_to_file(cross_correlations, "xcorr_results.txt")

def rsquared(theoretical_data, experimental_data):
    r_squared_arr = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data: 
            r_value = r2_score(theoretical_array, experimental_array)

            r_squared_arr.append(r_value)

    write_results_to_file(r_squared_arr, "rsquare_results.txt")

def pearsoncalc(theoretical_data, experimental_data):
    pearson_arr = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data:
            r, p_value = pearsonr(theoretical_array, experimental_array)

            pearson_arr.append(r)

    write_results_to_file(pearson_arr, "pearson_results.txt")

def dynamic_time_warping(theoretical_data, experimental_data):
    dtw_results = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data:
            # Convert lists to NumPy arrays and ensure they are 1D
            theoretical_array = np.array(theoretical_array, dtype=float).ravel()
            experimental_array = np.array(experimental_array, dtype=float).ravel()

            # Calculate DTW distance using the Euclidean distance function
            dtw_distance = euclidean(theoretical_array, experimental_array)

            # Append DTW distance to the results list
            dtw_results.append(dtw_distance)

    write_results_to_file(dtw_results, "dtw_results.txt")


def fourier_transform(data, filename):
    data_fft = np.abs(fft(data))

    write_results_to_file(data_fft.tolist(), filename)


def wavelet_transform(data, filename):
    wavelet_results = []
    formatted_wavelet_results = []

    for data_array in data:
        # Choose a wavelet family and level for the transform
        wavelet = "haar"
        level = 1

        # Perform the discrete wavelet transform
        coeffs = pywt.wavedec(data_array, wavelet, level=level)

        # Append the wavelet coefficients to the results list
        wavelet_results.append(coeffs)

        # Flatten the structure
        flattened_wavelet_results = [item for sublist in wavelet_results for item in sublist]

        # Convert the NumPy arrays to lists
        formatted_wavelet_results = [item.tolist() for item in flattened_wavelet_results]

    write_results_to_file(formatted_wavelet_results, filename)


def kmeans_clustering(theoretical_data, experimental_data):
    kmeans_results = []
    num_clusters = 3

    # Convert the data to a NumPy array
    data_array = experimental_data + theoretical_data

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, init=theoretical_data[1::2], n_init=1, random_state=0)
    kmeans.fit(data_array)                                        

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    for label in cluster_labels:
        kmeans_results.append(label)

    write_results_to_file(kmeans_results, "cluster_results.txt")


def kolmogorov_smirnov_test(theoretical_data, experimental_data):
    ks_results = []

    for theoretical_array in theoretical_data:
        for experimental_array in experimental_data:
            ks_statistic, ks_p_value = ks_2samp(theoretical_array, experimental_array)
            ks_results.append((ks_statistic, ks_p_value))

    ks_results_formatted = [[ks_statistic, ks_p_value] for ks_statistic, ks_p_value in ks_results]

    write_results_to_file(ks_results_formatted, "kstest_results.txt")


sanitised_data = import_sanitised_data()
normalized_sanitised_data = normalize_data(sanitised_data)

# Mieplot (2um 10dg, 2um 20dg, 4.8um 10dg, 4.8um 20dg, 9.6um 10dg, 9.6um 20dg)
theoretical_data = [[371.3715917, 319.0925216, 290.2519777, 287.0223039, 311.5366644, 365.8748561, 452.0489903, 571.9891332, 727.5290011, 920.3918037, 1152.176335, 1424.343408, 1738.202735, 2094.900348, 2495.406655, 2940.505232, 3430.78244, 3966.617946, 4548.17625, 5175.399281, 5848.000142, 6565.45807, 7327.014673, 8131.671486, 8978.188909, 9865.086544, 10790.64497, 11752.90896, 12749.69218, 13778.5833, 14836.95362, 15921.96603, 17030.58543, 18159.59049, 19305.58663, 20465.02038, 21634.19473, 22809.28571, 23986.35987, 25161.39268, 26330.28778, 27488.89684, 28633.03999, 29758.52677, 30861.17731, 31936.84378, 32981.43184, 33990.92201, 34961.39092, 35889.0321, 36770.17639, 37601.31165, 38379.10187, 39100.40526, 39762.29151, 40362.05784, 40897.244, 41365.64582, 41765.32748, 42094.63231, 42352.19201, 42536.93432, 42648.08901, 42685.19221, 42685.19221, 42648.08901, 42536.93432, 42352.19201, 42094.63231, 41765.32748, 41365.64582, 40897.244, 40362.05784, 39762.29151, 39100.40526, 38379.10187, 37601.31165, 36770.17639, 35889.0321, 34961.39092, 33990.92201, 32981.43184, 31936.84378, 30861.17731, 29758.52677, 28633.03999, 27488.89684, 26330.28778, 25161.39268, 23986.35987, 22809.28571, 21634.19473, 20465.02038, 19305.58663, 18159.59049, 17030.58543, 15921.96603, 14836.95362, 13778.5833, 12749.69218, 11752.90896, 10790.64497, 9865.086544, 8978.188909, 8131.671486, 7327.014673, 6565.45807, 5848.000142, 5175.399281, 4548.17625, 3966.617946, 3430.78244, 2940.505232, 2495.406655, 2094.900348, 1738.202735, 1424.343408, 1152.176335, 920.3918037, 727.5290011, 571.9891332, 452.0489903, 365.8748561, 311.5366644, 287.0223039, 290.2519777, 319.0925216, 371.3715917],
                    [554.2489401, 672.2223289, 816.0333381, 984.5003594, 1175.640383, 1386.664418, 1613.995685, 1853.312053, 2099.613663, 2347.31607, 2590.368552, 2822.396513, 3036.866085, 3227.268285, 3387.319251, 3511.172305, 3593.636894, 3630.398812, 3618.235568, 3555.220412, 3440.908266, 3276.496809, 3064.956099, 2811.120517, 2521.737399, 2205.467571, 1872.833992, 1536.115973, 1209.18779, 907.3020795, 646.8199721, 444.8916331, 319.0925216, 287.0223039, 365.8748561, 571.9891332, 920.3918037, 1424.343408, 2094.900348, 2940.505232, 3966.617946, 5175.399281, 6565.45807, 8131.671486, 9865.086544, 11752.90896, 13778.5833, 15921.96603, 18159.59049, 20465.02038, 22809.28571, 25161.39268, 27488.89684, 29758.52677, 31936.84378, 33990.92201, 35889.0321, 37601.31165, 39100.40526, 40362.05784, 41365.64582, 42094.63231, 42536.93432, 42685.19221, 42685.19221, 42536.93432, 42094.63231, 41365.64582, 40362.05784, 39100.40526, 37601.31165, 35889.0321, 33990.92201, 31936.84378, 29758.52677, 27488.89684, 25161.39268, 22809.28571, 20465.02038, 18159.59049, 15921.96603, 13778.5833, 11752.90896, 9865.086544, 8131.671486, 6565.45807, 5175.399281, 3966.617946, 2940.505232, 2094.900348, 1424.343408, 920.3918037, 571.9891332, 365.8748561, 287.0223039, 319.0925216, 444.8916331, 646.8199721, 907.3020795, 1209.18779, 1536.115973, 1872.833992, 2205.467571, 2521.737399, 2811.120517, 3064.956099, 3276.496809, 3440.908266, 3555.220412, 3618.235568, 3630.398812, 3593.636894, 3511.172305, 3387.319251, 3227.268285, 3036.866085, 2822.396513, 2590.368552, 2347.31607, 2099.613663, 1853.312053, 1613.995685, 1386.664418, 1175.640383, 984.5003594, 816.0333381, 672.2223289, 554.2489401],
                    [5326.675561, 6032.422108, 6706.846354, 7344.038848, 7948.615418, 8535.617106, 9129.531261, 9762.435138, 10471.32257, 11294.73465, 12268.87334, 13423.42876, 14777.39347, 16335.16627, 18083.2624, 19987.94311, 21994.05502, 24025.3282, 25986.32194, 27766.13062, 29243.8723, 30295.88296, 30804.43455, 30667.69044, 29810.51319, 28195.65196, 25834.76658, 22798.69627, 19226.35847, 15331.6691, 11407.91244, 7829.056819, 5047.60979, 3588.731529, 4040.472406, 7040.16563, 13257.1808, 23372.42196, 38055.1261, 57937.67636, 83589.28102, 115489.4764, 154002.4832, 199353.4774, 251607.8221, 310654.2492, 376192.8768, 447728.7991, 524571.8078, 605842.5837, 690485.4666, 777287.6552, 864904.4425, 951889.8399, 1036731.724, 1117890.434, 1193839.594, 1263107.819, 1324319.889, 1376235.979, 1417787.576, 1448108.805, 1466562.048, 1472756.941, 1472756.941, 1466562.048, 1448108.805, 1417787.576, 1376235.979, 1324319.889, 1263107.819, 1193839.594, 1117890.434, 1036731.724, 951889.8399, 864904.4425, 777287.6552, 690485.4666, 605842.5837, 524571.8078, 447728.7991, 376192.8768, 310654.2492, 251607.8221, 199353.4774, 154002.4832, 115489.4764, 83589.28102, 57937.67636, 38055.1261, 23372.42196, 13257.1808, 7040.16563, 4040.472406, 3588.731529, 5047.60979, 7829.056819, 11407.91244, 15331.6691, 19226.35847, 22798.69627, 25834.76658, 28195.65196, 29810.51319, 30667.69044, 30804.43455, 30295.88296, 29243.8723, 27766.13062, 25986.32194, 24025.3282, 21994.05502, 19987.94311, 18083.2624, 16335.16627, 14777.39347, 13423.42876, 12268.87334, 11294.73465, 10471.32257, 9762.435138, 9129.531261, 8535.617106, 7948.615418, 7344.038848, 6706.846354, 6032.422108, 5326.675561],
                    [5203.201665, 6388.653503, 7423.420999, 8159.614086, 8484.175487, 8340.202144, 7742.240746, 6782.123128, 5622.81212, 4479.347689, 3588.170073, 3168.519483, 3381.792126, 4296.131777, 5863.680912, 7916.528718, 10184.50397, 12333.95514, 14022.2525, 14958.86308, 14961.4387, 13995.1686, 12186.01923, 9803.201474, 7212.477924, 4808.487668, 2939.63059, 1841.820048, 1596.627672, 2124.783822, 3218.317848, 4605.31621, 6032.422108, 7344.038848, 8535.617106, 9762.435138, 11294.73465, 13423.42876, 16335.16627, 19987.94311, 24025.3282, 27766.13062, 30295.88296, 30667.69044, 28195.65196, 22798.69627, 15331.6691, 7829.056819, 3588.731529, 7040.16563, 23372.42196, 57937.67636, 115489.4764, 199353.4774, 310654.2492, 447728.7991, 605842.5837, 777287.6552, 951889.8399, 1117890.434, 1263107.819, 1376235.979, 1448108.805, 1472756.941, 1472756.941, 1448108.805, 1376235.979, 1263107.819, 1117890.434, 951889.8399, 777287.6552, 605842.5837, 447728.7991, 310654.2492, 199353.4774, 115489.4764, 57937.67636, 23372.42196, 7040.16563, 3588.731529, 7829.056819, 15331.6691, 22798.69627, 28195.65196, 30667.69044, 30295.88296, 27766.13062, 24025.3282, 19987.94311, 16335.16627, 13423.42876, 11294.73465, 9762.435138, 8535.617106, 7344.038848, 6032.422108, 4605.31621, 3218.317848, 2124.783822, 1596.627672, 1841.820048, 2939.63059, 4808.487668, 7212.477924, 9803.201474, 12186.01923, 13995.1686, 14961.4387, 14958.86308, 14022.2525, 12333.95514, 10184.50397, 7916.528718, 5863.680912, 4296.131777, 3381.792126, 3168.519483, 3588.170073, 4479.347689, 5622.81212, 6782.123128, 7742.240746, 8340.202144, 8484.175487, 8159.614086, 7423.420999, 6388.653503, 5203.201665],
                    [26292.81129, 19670.0198, 15102.36738, 12802.26955, 12592.83697, 14026.18127, 16560.413, 19745.11102, 23360.85514, 27468.85052, 32350.16331, 38345.43917, 45637.37953, 54041.06283, 62874.13577, 70966.10844, 76834.44973, 79011.02366, 76455.92235, 68959.20487, 57415.74772, 43871.62965, 31283.0598, 22994.15721, 22014.43527, 30242.8453, 47824.47738, 72824.36521, 101354.3239, 128198.1719, 147863.9695, 155873.7626, 150010.5339, 131205.1614, 103780.7973, 74881.56093, 53083.03512, 46383.7033, 59967.38534, 94261.24982, 143852.2322, 197742.2015, 241217.3356, 259304.3835, 241434.6047, 186601.0303, 108047.5952, 36432.40554, 20502.44788, 124611.1286, 422873.0653, 990316.7125, 1891972.392, 3171317.568, 4839796.517, 6869167.508, 9188177.771, 11684544.3, 14212495.25, 16605309.65, 18691512.12, 20312763.75, 21341148.15, 21693547.44, 21693547.44, 21341148.15, 20312763.75, 18691512.12, 16605309.65, 14212495.25, 11684544.3, 9188177.771, 6869167.508, 4839796.517, 3171317.568, 1891972.392, 990316.7125, 422873.0653, 124611.1286, 20502.44788, 36432.40554, 108047.5952, 186601.0303, 241434.6047, 259304.3835, 241217.3356, 197742.2015, 143852.2322, 94261.24982, 59967.38534, 46383.7033, 53083.03512, 74881.56093, 103780.7973, 131205.1614, 150010.5339, 155873.7626, 147863.9695, 128198.1719, 101354.3239, 72824.36521, 47824.47738, 30242.8453, 22014.43527, 22994.15721, 31283.0598, 43871.62965, 57415.74772, 68959.20487, 76455.92235, 79011.02366, 76834.44973, 70966.10844, 62874.13577, 54041.06283, 45637.37953, 38345.43917, 32350.16331, 27468.85052, 23360.85514, 19745.11102, 16560.413, 14026.18127, 12592.83697, 12802.26955, 15102.36738, 19670.0198, 26292.81129],
                    [13655.82288, 17379.95705, 23447.01474, 28562.0258, 29731.14768, 26265.76047, 20407.7521, 16021.06504, 16118.90789, 20872.8672, 27476.94729, 31955.93601, 31653.88955, 26736.20868, 19857.18032, 14477.58487, 13093.27705, 16346.65614, 23095.40116, 30929.25109, 36812.58947, 38035.09843, 33635.38152, 25696.89107, 19209.64122, 19656.9196, 29253.9855, 44470.97839, 57281.54506, 60028.02876, 50674.33611, 34342.92258, 19670.0198, 12802.26955, 14026.18127, 19745.11102, 27468.85052, 38345.43917, 54041.06283, 70966.10844, 79011.02366, 68959.20487, 43871.62965, 22994.15721, 30242.8453, 72824.36521, 128198.1719, 155873.7626, 131205.1614, 74881.56093, 46383.7033, 94261.24982, 197742.2015, 259304.3835, 186601.0303, 36432.40554, 124611.1286, 990316.7125, 3171317.568, 6869167.508, 11684544.3, 16605309.65, 20312763.75, 21693547.44, 21693547.44, 20312763.75, 16605309.65, 11684544.3, 6869167.508, 3171317.568, 990316.7125, 124611.1286, 36432.40554, 186601.0303, 259304.3835, 197742.2015, 94261.24982, 46383.7033, 74881.56093, 131205.1614, 155873.7626, 128198.1719, 72824.36521, 30242.8453, 22994.15721, 43871.62965, 68959.20487, 79011.02366, 70966.10844, 54041.06283, 38345.43917, 27468.85052, 19745.11102, 14026.18127, 12802.26955, 19670.0198, 34342.92258, 50674.33611, 60028.02876, 57281.54506, 44470.97839, 29253.9855, 19656.9196, 19209.64122, 25696.89107, 33635.38152, 38035.09843, 36812.58947, 30929.25109, 23095.40116, 16346.65614, 13093.27705, 14477.58487, 19857.18032, 26736.20868, 31653.88955, 31955.93601, 27476.94729, 20872.8672, 16118.90789, 16021.06504, 20407.7521, 26265.76047, 29731.14768, 28562.0258, 23447.01474, 17379.95705, 13655.82288]
                    ]

# Turn linear to logarithmic data because of extreme peaks in the middle
theoretical_data = [[math.log(element) for element in sub_array] for sub_array in theoretical_data]

normalized_theoretical_data = normalize_data(theoretical_data)

#coherence(normalized_theoretical_data, normalized_sanitised_data)
#cross_correlation(normalized_theoretical_data, normalized_sanitised_data)
#dynamic_time_warping(normalized_theoretical_data, normalized_sanitised_data)
#fourier_transform(normalized_theoretical_data, "fourier_theoretical_results.txt")
#fourier_transform(normalized_sanitised_data, "fourier_experimental_results.txt")
#wavelet_transform(normalized_theoretical_data, "wavelet_theoretical_results.txt")
#wavelet_transform(normalized_sanitised_data, "wavelet_experimental_results.txt")
#kmeans_clustering(normalized_theoretical_data, normalized_sanitised_data)
#kolmogorov_smirnov_test(normalized_theoretical_data, normalized_sanitised_data)
rsquared(normalized_theoretical_data, normalized_sanitised_data)
