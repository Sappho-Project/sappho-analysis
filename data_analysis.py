import numpy as np
from scipy import signal
import os
from scipy.spatial.distance import euclidean
from scipy.fft import fft
import pywt
from sklearn.cluster import KMeans
from scipy.stats import ks_2samp


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
            f, Pxy = signal.csd(theoretical_array, experimental_array, fs=0.1, nperseg=16)
            f, Pxx = signal.csd(theoretical_array, theoretical_array, fs=0.1, nperseg=16)
            f, Pyy = signal.csd(experimental_array, experimental_array, fs=0.1, nperseg=16)

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
    combined_data = theoretical_data + experimental_data
    kmeans_results = []
    num_clusters = 2

    # Convert the data to a NumPy array
    data_array = np.array(combined_data)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
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

# Replace with actual theoretical data
theoretical_data = [[1030.5, 1021.2, 1014.6, 1092.7, 1057.9, 1028.1, 1120.0, 1255.9, 1258.5, 1262.3, 1241.9, 1276.5, 1233.7, 1316.5, 1381.5, 1400.1, 1429.3, 1432.7, 1555.8, 1606.0, 1615.2, 1610.4, 1921.6, 2016.6, 2169.1, 2189.5, 2244.9, 2632.3, 2857.8, 3062.0, 3654.4, 3938.3, 3938.1, 3937.4, 3938.3, 3937.3, 3938.7, 3939.2, 3937.9, 3938.0, 3937.5, 3938.0, 3938.2, 3938.9, 3938.1, 3938.3, 3937.1, 3938.0, 3937.1, 3938.6, 3937.5, 3938.6, 3937.6, 3939.2, 3937.9, 3938.8, 3938.4, 3938.3, 3939.1, 3938.5, 3938.7, 3936.7, 3938.6, 3937.1, 3938.1, 3936.8, 3938.1, 3937.2, 3938.4, 3937.3, 3938.9, 3937.8, 3938.1, 3937.7, 3937.9, 3937.9, 3938.1, 3938.6, 3938.6, 3937.7, 3937.5, 3937.9, 3937.9, 3881.9, 3408.5, 2639.0, 2404.8, 2254.7, 2124.0, 1894.6, 1877.3, 1595.9, 1462.6, 1440.4, 1505.4, 1372.7, 1305.5, 1214.2, 1220.5, 1213.3, 1155.9, 1222.4, 1166.9, 1117.7, 1107.5, 1030.9, 1049.6, 1059.2, 1059.5, 1078.4, 1044.5, 913.4, 990.3, 923.0, 909.8, 921.6, 949.0, 929.7, 945.1, 892.3, 877.8, 954.1, 900.6, 855.7, 940.8, 868.7, 873.5, 824.8],
                    [685.4, 698.5666666666667, 686.9333333333333, 698.7, 696.8333333333334, 694.0, 714.2, 710.2, 713.6666666666666, 719.6333333333333, 720.4, 723.8, 735.8666666666667, 726.9666666666667, 743.4333333333333, 733.0666666666667, 759.7, 745.7666666666667, 775.0333333333333, 765.6, 759.6, 753.8333333333334, 784.0333333333333, 733.2666666666667, 778.3333333333334, 763.9333333333333, 780.3, 740.4, 754.4, 772.1, 765.8333333333334, 798.3, 828.5666666666667, 791.5333333333333, 828.2666666666667, 806.5666666666667, 798.9666666666667, 832.4333333333333, 809.7333333333333, 758.0666666666667, 769.0333333333333, 816.8666666666667, 838.2, 845.9, 830.3, 837.2, 831.5666666666667, 844.7, 851.2333333333333, 832.6, 852.7666666666667, 822.4333333333333, 837.9333333333333, 825.5, 836.3, 830.7666666666667, 843.5, 833.7333333333333, 823.7333333333333, 807.7, 836.7, 868.2333333333333, 828.9333333333333, 835.2, 822.9666666666667, 819.8333333333334, 765.6, 839.1666666666666, 824.6, 742.3666666666667, 779.9333333333333, 820.7333333333333, 813.5666666666667, 810.7, 801.0333333333333, 801.7333333333333, 786.6, 803.8, 754.1, 749.4, 787.2666666666667, 789.4333333333333, 771.0666666666667, 768.0333333333333, 728.3333333333334, 703.0666666666667, 761.0333333333333, 743.4, 738.2666666666667, 735.4666666666667, 735.8666666666667, 735.8666666666667, 738.5333333333333, 726.4666666666667, 730.2, 716.1666666666666, 716.3333333333334, 675.5666666666667, 707.8666666666667, 691.6, 673.6333333333333, 691.4, 677.9333333333333, 670.9333333333333, 669.8666666666667, 656.2333333333333, 662.8, 658.2666666666667, 656.6, 650.2666666666667, 653.6666666666666, 629.0, 642.8333333333334, 635.9333333333333, 629.4333333333333, 628.7, 624.1666666666666, 599.4333333333333, 588.6, 600.2, 598.3666666666667, 573.4666666666667, 596.6, 584.0666666666667, 579.5, 570.7, 553.0, 573.0666666666667]]
normalized_theoretical_data = normalize_data(theoretical_data)

coherence(normalized_theoretical_data, normalized_sanitised_data)
cross_correlation(normalized_theoretical_data, normalized_sanitised_data)
dynamic_time_warping(normalized_theoretical_data, normalized_sanitised_data)
fourier_transform(normalized_theoretical_data, "fourier_theoretical_results.txt")
fourier_transform(normalized_sanitised_data, "fourier_experimental_results.txt")
wavelet_transform(normalized_theoretical_data, "wavelet_theoretical_results.txt")
wavelet_transform(normalized_sanitised_data, "wavelet_experimental_results.txt")
kmeans_clustering(normalized_theoretical_data, normalized_sanitised_data)
kolmogorov_smirnov_test(normalized_theoretical_data, normalized_sanitised_data)
