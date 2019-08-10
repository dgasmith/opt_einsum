import argparse
import json
import os
import shutil
import subprocess as sp

# Args
parser = argparse.ArgumentParser(description='Creates a conda environment from file for a given Python version.')
parser.add_argument('-n', '--name', type=str, nargs=1,
                    help='The name of the created Python environment')
parser.add_argument('-p', '--python', type=str, nargs=1,
                    help='The version of the created Python environment')
parser.add_argument('conda_file', nargs='*',
                    help='The file for the created Python environment')

args = parser.parse_args()

with open(args.conda_file[0], "r") as handle:
    script = handle.read()

tmp_file = "tmp_env.yaml"
script = script.replace("- python", "- python {}*".format(args.python[0]))

with open(tmp_file, "w") as handle:
    handle.write(script)

# Figure out conda path
if "CONDA_EXE" in os.environ:
    conda_path = os.environ["CONDA_EXE"]
else:
    conda_path = shutil.which("conda")

print("CONDA ENV NAME  {}".format(args.name[0]))
print("PYTHON VERSION  {}".format(args.python[0]))
print("CONDA FILE NAME {}".format(args.conda_file[0]))
print("CONDA path      {}".format(conda_path))

sp.call("{} env create -n {} -f {}".format(conda_path, args.name[0], tmp_file), shell=True)
os.unlink(tmp_file)
