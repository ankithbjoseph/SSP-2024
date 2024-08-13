import subprocess


def run_script(script_name, *args):
    command = ["python", script_name] + list(args)
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout)
    print(result.stderr)


if __name__ == "__main__":
    run_script("runner1.py", "--input_file", "k4.txt", "--k", "4", "--runner", "inline")
    run_script("runner1.py", "--input_file", "k4.txt", "--k", "4", "--runner", "inline")
    run_script("runner2.py", "--input_file", "k4.txt", "--k", "4", "--runner", "inline")
    run_script("runner2.py", "--input_file", "k4.txt", "--k", "4", "--runner", "inline")
