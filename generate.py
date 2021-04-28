import argparse
import os
import shlex
import subprocess

from tqdm import tqdm


# python generate.py 100 --unique_inputs 2 --inputs 4 --functions 5 --num_samples_per_comp 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("number", type=int, default=10000, help="Number of samples")

    parser.add_argument(
        "--functions",
        type=int,
        default=5,
        help="Maximum number of functions to be used in the sample",
    )
    parser.add_argument(
        "--io", type=int, default=1, help="Number of io examples per sample"
    )
    parser.add_argument(
        "--inputs", type=int, default=2, help="Number of input lists in each io example"
    )
    parser.add_argument(
        "--unique_inputs",
        type=int,
        default=1,
        help="Number of unique input lists provided",
    )
    parser.add_argument(
        "--num_samples_per_comp",
        type=int,
        default=1,
        help="Number of samples to generate from a single Composition",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="dataset.dat",
        help="Name of the file to save to",
    )
    parser.add_argument(
        "--train", action="store_true", help="Whether to generate training examples"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of processes that should be used while generating",
    )

    parser.add_argument("--test", action="store_true", help="Generate test dataset")

    return parser.parse_args()


args = parse_args()


def increment_progressbar(processes, can_read, t):
    for i, proc in enumerate(processes):
        line = proc.stdout.readline()
        if not line:
            can_read[i] = False
        else:
            t.update()
    return any(can_read)


def main():
    cpu_count = args.num_processes
    total_per_non_last_thread = args.number // cpu_count
    total_last_thread = args.number - total_per_non_last_thread * (cpu_count - 1)

    processes = []
    files = []

    cmd_string = (
        f"python -u -m src.generate_utils sample_number fname --unique_inputs {args.unique_inputs} "
        f"--inputs {args.inputs} --functions {args.functions} --num_samples_per_comp "
        f"{args.num_samples_per_comp} --io {args.io} {'--test' if args.test else ''}"
    )

    for i in range(cpu_count - 1):
        fname = f"temp_dataset_{i}.dat"
        files.append(fname)
        curr = cmd_string.replace("fname", fname, 1).replace(
            "sample_number", str(total_per_non_last_thread), 1
        )
        process = subprocess.Popen(shlex.split(curr), stdout=subprocess.PIPE,)
        processes.append(process)
    fname = f"temp_dataset_{cpu_count-1}.dat"
    files.append(fname)
    curr = cmd_string.replace("fname", fname, 1).replace(
        "sample_number", str(total_last_thread), 1
    )

    process = subprocess.Popen(shlex.split(curr), stdout=subprocess.PIPE)
    processes.append(process)

    progressbar = tqdm(total=args.number * args.num_samples_per_comp)

    can_read = [True] * len(processes)
    try:
        while True:
            result = increment_progressbar(processes, can_read, progressbar)
            if not result:
                break
    except KeyboardInterrupt:
        for proc in processes:
            proc.kill()
        raise

    for proc in processes:
        proc.wait()

    if not os.path.isdir("datasets"):
        os.mkdir("datasets")

    with open(f"datasets/{args.file_name}", "a") as f:
        f.truncate(0)
        for fname in files:
            with open(fname) as g:
                f.writelines(g.readlines())
            os.remove(fname)

    progressbar.close()


if __name__ == "__main__":
    main()
