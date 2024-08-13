#!/usr/bin/env python

from mrjob.job import MRJob, MRStep
import json
import math


class KMeans(MRJob):
    centroids = []

    def steps(self):
        return [
            MRStep(
                mapper_init=self.load_options,
                mapper=self.mapper,
                reducer=self.reducer,
            )
        ]

    def configure_args(self):
        super(KMeans, self).configure_args()
        self.add_file_arg("--centroids")

    def load_options(self):
        if self.options.centroids:
            with open(self.options.centroids, "r") as f:
                self.centroids = json.load(f)

    def euclidean_distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def mapper(self, _, line):
        observation = [float(x) for x in line.strip().split(",")]
        distances = [
            self.euclidean_distance(observation, centroid)
            for centroid in self.centroids
        ]
        closest = distances.index(min(distances))
        yield closest, observation

    def reducer(self, key, values):
        points = list(values)
        new_centroid = [sum(x) / len(x) for x in zip(*points)]
        yield key, new_centroid


if __name__ == "__main__":
    KMeans.run()
