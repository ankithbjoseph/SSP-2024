from mrjob.job import MRJob
import numpy as np
import random

class MRKMeansPlusPlus(MRJob):

    def configure_args(self):
        super(MRKMeansPlusPlus, self).configure_args()
        self.add_passthru_arg('--k', type=int, help='Number of centroids')

    def mapper(self, _, line):
        data_point = [float(x) for x in line.split(',')]
        yield None, data_point

    def combiner(self, key, values):
        data_points = list(values)
        if len(data_points) == 0:
            return

        # Select the first centroid randomly
        centroids = [random.choice(data_points)]

        for _ in range(1, self.options.k):
            distances = np.array([min(np.linalg.norm(np.array(dp) - np.array(c)) for c in centroids) for dp in data_points])
            probabilities = distances / distances.sum()
            new_centroid = data_points[np.random.choice(len(data_points), p=probabilities)]
            centroids.append(new_centroid)

        for centroid in centroids:
            yield None, centroid

    def reducer(self, key, values):
        data_points = list(values)
        if len(data_points) == 0:
            return

        # Select the first centroid randomly
        centroids = [random.choice(data_points)]

        for _ in range(1, self.options.k):
            distances = np.array([min(np.linalg.norm(np.array(dp) - np.array(c)) for c in centroids) for dp in data_points])
            probabilities = distances / distances.sum()
            new_centroid = data_points[np.random.choice(len(data_points), p=probabilities)]
            centroids.append(new_centroid)

        for centroid in centroids:
            yield None, centroid

if __name__ == '__main__':
    MRKMeansPlusPlus.run()
