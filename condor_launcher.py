import os
import shutil
from glob import glob
import subprocess
import shlex


INTERPRETER_PATH = "/esat/opal/kkelchte/conda/envs/venv/bin/python"
PROJECT_PATH = "/users/visics/kkelchte/code/contrastive-learning"
OUTPUT_PATH = "/esat/opal/kkelchte/experimental_data/contrastive_learning"

SPECS = {
    "Universe": "vanilla",
    "Requirements": '(CUDAGlobalMemoryMb >= 3900)',
    "initial_dir": PROJECT_PATH,
    "priority": 1,
    "RequestCpus": 6,
    "Request_GPUs": 1,
    "RequestMemory": "5 G",
    "RequestDisk": "50 G",
    "Niceuser": "True",
    "+RequestWalltime": 60 * 60 * 2,
}

TARGETS = ["cone", "gate", "line"]
CONFIGS = [f"configs/{cf}.json" for cf in ["baseline", "augment", "augment_blur", "augment_blur_triplet"]]
LEARNING_RATES = [0.01, 0.001, 0.0001, 0.00001]

# TARGETS = ["cone"]
# CONFIGS = [f"configs/{cf}.json" for cf in ["baseline"]]
# LEARNING_RATES = [0.01]

SUBMIT = True
RM_EXIST = True

print("TARGETS: ", TARGETS)
print("CONFIGS: ", CONFIGS)
print("LEARNING_RATES: ", LEARNING_RATES)


def create_condor_job_file(trgt, config, lrate):
    config_tag = os.path.basename(config[:-5])
    output_dir = (
        f"{OUTPUT_PATH}/{trgt}/{config_tag}/{lrate}"
    )
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} \n")
        jobfile.write(
            f"arguments     = {PROJECT_PATH}/run.py --config_file {PROJECT_PATH}/{config} "
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
