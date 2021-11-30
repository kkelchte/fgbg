import os
import time
import shutil
import subprocess
import shlex


INTERPRETER_PATH = "/esat/opal/kkelchte/conda/envs/venv/bin/python"
PROJECT_PATH = "/users/visics/kkelchte/code/contrastive-learning"

SPECS = {
    "Universe": "vanilla",
    "Requirements": "(CUDAGlobalMemoryMb >= 3900) && (CUDACapability < 8.6)",
    "initial_dir": PROJECT_PATH,
    "priority": 1,
    "RequestCpus": 4,
    "Request_GPUs": 1,
    "RequestMemory": "10 G",
    "RequestDisk": "50 G",
    "Niceuser": "True",
    "+RequestWalltime": int(50 * 3 * 60),
}

# RED_LINE
TARGET = "red_line"
CONFIG = "deep_supervision_only_brightness"

# GATE
# TARGET = 'gate'
# CONFIG = 'deep_supervision_comb_blur_brightness_hue_bn'

ENCODER = f"data/{TARGET}/{CONFIG}"
TASKS = ["waypoints", "velocities"]
LEARNING_RATE = 0.00001
SUBMIT = True
RM_EXIST = True
NUMEPOCH = 50


def create_condor_job_file(trgt, task, lrate):
    output_dir = f"data/{trgt}/{CONFIG}/{task}"
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} \n")
        jobfile.write(
            f"arguments = {PROJECT_PATH}/run.py --config_file "
            f"{PROJECT_PATH}/configs/{CONFIG}.json "
            f"--learning_rate {lrate} --target {trgt} "
            f"--output_dir {output_dir} "
            f"--encoder_ckpt_dir {ENCODER} "
            f"--task {task} --number_of_epochs {NUMEPOCH} \n"
        )

        for key, value in SPECS.items():
            jobfile.write(f"{key} \t = {value} \n")

        jobfile.write(f"error   = {output_dir}/condor.err\n")
        jobfile.write(f"output  = {output_dir}/condor.out\n")
        jobfile.write(f"log     = {output_dir}/condor.log\n")
        jobfile.write("Queue \n")

    return os.path.join(output_dir, "condor.job")


for task in TASKS:
    filename = create_condor_job_file(TARGET, task, LEARNING_RATE)
    if SUBMIT:
        print(f"submitting {filename}")
        subprocess.call(shlex.split(f"condor_submit {filename}"))
        time.sleep(0.1)

print("finished")
