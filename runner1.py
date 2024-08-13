import logging
import json
from selectrandom import SelectKRandomInstances
from mrkmeans import KMeans
import argparse

## setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("output.log")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def run_select_k_random_instances(input_file, k, initial_centroids_file, runner):
    logger.info("Starting MRJob to select %d random instances from %s", k, input_file)

    # Configure the MRJob runner
    mr_job = SelectKRandomInstances(
        args=[input_file, "--k", str(k), f"--runner={runner}"]
    )

    # Run the job and capture output
    with mr_job.make_runner() as runner:
        runner.run()

        logger.info(
            "Processing results and saving initial centroids to %s",
            initial_centroids_file,
        )

        centroids = []
        with open(initial_centroids_file, "w") as f_out:
            for key, value in mr_job.parse_output(runner.cat_output()):
                centroids.append(value)

            # Dump the centroids as JSON
            json.dump(centroids, f_out, indent=4)

        logger.info(
            "Successfully saved initial centroids to %s", initial_centroids_file
        )
        return centroids


def run_kmeans(input_file, centroids_file, iterations, runner):
    logger.info(
        "Starting MRJob for K-Means clustering using centroids from %s", centroids_file
    )

    with open(centroids_file, "r") as f:
        centroids = json.load(f)

    # Configure the MRJob runner for K-Means
    mr_job = KMeans(
        args=[
            input_file,
            "--centroids",
            centroids_file,
            f"--runner={runner}",
        ]
    )

    # Run the K-Means job and capture output
    i = 0
    while i < iterations:
        logger.info(f"Iteration {i}")
        with mr_job.make_runner() as runner:
            runner.run()
            new_centroids = []

            # print(f"Centroids: {centroids}")

            for key, value in mr_job.parse_output(runner.cat_output()):
                new_centroids.append(value)

            with open(centroids_file, "w") as f:
                json.dump(new_centroids, f)

            logger.info(f"New Centroids: {new_centroids}")
            if sorted(centroids) == sorted(new_centroids):
                logger.info("Centroids have stabilized. Exiting loop.")
                break

            centroids = new_centroids

        i += 1

    return centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run K-Means Clustering with initial centroids."
    )

    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to your input file"
    )
    parser.add_argument(
        "--k", type=int, required=True, help="Number of initial centroids to select"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of K-Means iterations"
    )
    parser.add_argument("--runner", type=str, default="hadoop", help="Runner for MRjob")

    args = parser.parse_args()

    logger.info(f"""######################### START #########################
                
    RANDOM INITIALIZATION  file={args.input_file} , k={args.k} , runner={args.runner} , iterations={args.iterations}     
                """)

    centroids_file = "centroids.json"  # Temp file to store initial centroids

    # Step 1: Run the SelectKRandomInstances job to generate initial centroids
    initial_centroids = run_select_k_random_instances(
        args.input_file, args.k, centroids_file, args.runner
    )

    # Step 2: Run the MRKMeans job to perform K-Means clustering and calculate DBI
    final_centroids = run_kmeans(
        args.input_file, centroids_file, args.iterations, args.runner
    )

    logger.info("#########################  END  #########################")
