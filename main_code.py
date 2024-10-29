import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import KMeans
import time

num_sub_clusters = 10
num_large_clusters = 1000


def nearest_neighbor(distance_matrix, current_city):
    total_distance = 0
    visited_cities = [current_city]
    for i in range(len(distance_matrix) - 1):
        distance_matrix[current_city, visited_cities] = np.inf
        nearest_city = np.argmin(distance_matrix[current_city])
        total_distance += distance_matrix[current_city, nearest_city]
        current_city = nearest_city
        visited_cities.append(current_city)
    total_distance += distance_matrix[
        current_city, visited_cities[0]]  # Add the distance from the final point to the starting point
    visited_cities.append(visited_cities[0])  # Append the starting point as the last element of the visited cities list
    return total_distance, visited_cities


def main():
    start_time_whole = time.time()
    file = "/Users/atchyutram/Downloads/cities.csv"
    data = open(file).readlines()
    new_data = []
    for line in data:
        line = line.strip().split(" ")
        new_data.append(line)

    df = pd.DataFrame(new_data, columns=["S.NO", 'X', 'Y'])
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])

    # Our process of Kmeans Clustering begins here
    large_cluster_start_time = time.time()

    kmeans_cluster = KMeans(n_clusters=num_large_clusters, random_state=10)
    coordinates = df[['X', 'Y']].values
    kmeans_cluster.fit(coordinates)
    df['LargeCluster'] = kmeans_cluster.labels_
    large_cluster_end_time = time.time()
    print(f'Time taken for large clustering: {large_cluster_end_time - large_cluster_start_time}')

    sub_cluster_start_time = time.time()
    for large_cluster in range(num_large_clusters):
        large_cluster_indices = df[
            df['LargeCluster'] == large_cluster].index.tolist()  # Data points are not fetched, only indices are fetched
        large_cluster_coordinates = df.loc[large_cluster_indices, ['X', 'Y']].values  # Data points are fetched
        # if len(large_cluster_coordinates) > num_sub_clusters:
        kmeans_sub_cluster = (KMeans(n_clusters=num_sub_clusters, random_state=0))
        kmeans_sub_cluster.fit(large_cluster_coordinates)
        df.loc[large_cluster_indices, 'SubCluster'] = kmeans_sub_cluster.labels_
    sub_cluster_end_time = time.time()
    print(f'Time taken for sub clustering: {sub_cluster_end_time - sub_cluster_start_time}')

    city_input = input("Enter the indices of the cities you want to visit, separated by commas: ")
    select_current_city = int(input("Enter the index of the city you want to start from: "))
    city_indices = list(map(int, city_input.split(',')))

    if select_current_city not in city_indices:
        city_indices.append(select_current_city)

    selected_df = df.loc[city_indices]

    grouped = selected_df.groupby('LargeCluster')

    cluster_grouping_start_time = time.time()
    final_route = []
    final_distance = 0
    for cluster, group in grouped:
        sub_grouped = group.groupby('SubCluster')
        for sub_cluster, sub_group in sub_grouped:
            indices = sub_group.index.tolist()
            coordinates = sub_group[['X', 'Y']].values

            distances = pdist(coordinates, metric='euclidean')
            dist_matrix = squareform(distances)

            if select_current_city in indices:
                start_city = indices.index(select_current_city)  # Dealing with indices of the cities
            else:
                start_city = 0

            distance, route = nearest_neighbor(dist_matrix, start_city)

            cluster_route = [indices[i] for i in route]  # Convert the indices to the actual city indices

            final_route.extend(cluster_route[:-1])
            final_distance += distance  # Till here, we found the routes and distances in the sub-clusters and large clusters
    cluster_grouping_end_time = time.time()
    print(f'Time taken for cluster grouping: {cluster_grouping_end_time - cluster_grouping_start_time}')

    start_time_dist = time.time()
    final_coordinates = df.loc[final_route][['X', 'Y']].values
    final_distances = pdist(final_coordinates, metric='euclidean')
    final_dist_matrix = squareform(final_distances)
    end_time_dist = time.time()
    print(f"The total time taken for the distance matrix: {end_time_dist - start_time_dist}")

    adjusted_start_city = final_route.index(select_current_city)
    total_distance, final_route_indices = nearest_neighbor(final_dist_matrix, adjusted_start_city)
    final_route_converted = [final_route[i] for i in final_route_indices]

    print(f'Final total distance: {total_distance}')
    print(f'Final route: {final_route_converted}')

    end_time_whole = time.time()
    print(f'Time taken for the whole process: {end_time_whole - start_time_whole}')


main()

