import numpy as np
import pandas as pd
import sys
import os

# Set the random seed for reproducibility
np.random.seed(42)

centers_4 = np.array(
    [
        [-2.50919762, 9.01428613, 4.63987884, 1.97316968, -6.87962719, -6.88010959, -8.838327],
        [-9.58831011, 9.39819704, 6.64885282, -5.75321779, -6.36350066, -6.3319098, -3.915155],
        [2.23705789, -7.21012279, -4.15710703, -2.67276313, -0.87860032, 5.70351923, -6.006524],
        [2.15089704, -6.58951753, -8.69896814, 8.97771075, 9.31264066, 6.16794696, -3.907724],
    ]
)

def generate_and_save_data(centers, num_samples_per_cluster, std_dev, batch_size, filename):
    num_clusters = len(centers)
    with open(filename, 'w') as f:
        for i, center in enumerate(centers):
            total_samples = 0
            while total_samples < num_samples_per_cluster:
                # Determine the size of the current batch
                current_batch_size = min(batch_size, num_samples_per_cluster - total_samples)
                cluster_data = np.random.normal(
                    loc=center, scale=std_dev, size=(current_batch_size, centers.shape[1])
                )
                
                # Convert the batch to DataFrame and write to file
                pd.DataFrame(cluster_data).to_csv(f, header=False, index=False, mode='a')
                
                # Update the number of samples generated so far
                total_samples += current_batch_size
                
                # Memory management
                del cluster_data

def main(num_clusters):
    # Parameters
    num_samples_per_cluster = 205 * 1000 * num_clusters
    std_dev = 1.0  # Standard deviation of the clusters
    batch_size = 10000  # Adjust the batch size according to your system's memory

    # Output filename
    filename = f"{num_clusters}_k4_large.txt"

    # Generate and save data for the 4-cluster dataset
    generate_and_save_data(centers_4, num_samples_per_cluster, std_dev, batch_size, filename)

    # Print the size of the file created
    file_size = os.path.getsize(filename)
    print(f"Size of the file '{filename}' created: {file_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generatedata.py <num_clusters>")
    else:
        num_clusters = int(sys.argv[1])
        main(num_clusters)
