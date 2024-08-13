from mrjob.job import MRJob
from mrjob.step import MRStep
from random import uniform
import heapq


class SelectKRandomInstances(MRJob):
    def configure_args(self):
        super(SelectKRandomInstances, self).configure_args()
        self.add_passthru_arg(
            "--k", type=int, default=4, help="Number of random instances to select"
        )

    def mapper_init(self):
        self.priority_queue = []
        self.k = self.options.k

    def mapper(self, _, line):
        priority = uniform(0, 1000)
        line_data = [float(x) for x in line.rstrip().split(",")]
        if len(self.priority_queue) < self.k:
            heapq.heappush(self.priority_queue, (priority, line_data))
        elif self.priority_queue[0][0] < priority:
            heapq.heapreplace(self.priority_queue, (priority, line_data))

    def mapper_final(self):
        for priority, line_data in self.priority_queue:
            yield None, (priority, line_data)

    def reducer_init(self):
        self.priority_queue = []
        self.k = self.options.k

    def reducer(self, _, priority_line_pairs):
        for priority, line_data in priority_line_pairs:
            if len(self.priority_queue) < self.k:
                heapq.heappush(self.priority_queue, (priority, line_data))
            elif self.priority_queue[0][0] < priority:
                heapq.heapreplace(self.priority_queue, (priority, line_data))

        for priority, line_data in self.priority_queue:
            yield None, line_data


if __name__ == "__main__":
    SelectKRandomInstances.run()
