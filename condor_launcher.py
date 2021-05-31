import os
import shutil
from glob import glob
import subprocess
import shlex


INTERPRETER_PATH = "/PATH/TO/INTERPRETER"
PROJECT_PATH = "/PATH/TO/PROJECT"

SPECS = {
    "Universe": "vanilla",
    "Requirements": '(machineowner == "Visics") && (CUDAGlobalMemoryMb >= 1900)',
    "priority": 1,
    "RequestCpus": 4,
    "Request_GPUs": 1,
    "RequestMemory": "4000 G",
    "RequestDisk": "50 G",
    "Niceuser": "True",
    "+RequestWalltime": 60 * 60,
}

TARGETS = ["cone", "gate", "line"]
CONFIGS = glob(f"{PROJECT_PATH}/configs/*.json")
LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001]
SUBMIT = False
RM_EXIST = False

print("TARGETS: ", TARGETS)
print("CONFIGS: ", CONFIGS)
print("LEARNING_RATES: ", LEARNING_RATES)


def create_condor_job_file(trgt, config, lrate):
    config_tag = os.path.basename(config[:-5])
    output_dir = (
        f"/esat/opal/kkelchte/experimental_data/fgbg/{trgt}/{config_tag}/{lrate}"
    )
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} {PROJECT_PATH}/run.py \n")
        jobfile.write(
            f"arguments     = --config_file {PROJECT_PATH}/{config} "
            f"--learning_rate {lrate} --target {trgt} "
            f"--output_dir {output_dir}\n"
        )

        for key, value in SPECS.items():
            jobfile.write(f"{key} \t = {value} \n")

        jobfile.write(f"error   = {output_dir}/condor.err\n")
        jobfile.write(f"output  = {output_dir}/condor.out\n")
        jobfile.write(f"log     = {output_dir}/condor.log\n")
        jobfile.write("Queue \n")

    return os.path.join(output_dir, "condor.job")


for target in TARGETS:
    for conf in CONFIGS:
        for lr in LEARNING_RATES:
            filename = create_condor_job_file(target, conf, lr)
            if SUBMIT:
                subprocess.call(shlex.split(f"condor_submit {filename}"))

print("finished")
