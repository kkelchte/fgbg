import os
import time
import shutil
import subprocess
import shlex


INTERPRETER_PATH = "/esat/opal/kkelchte/conda/envs/venv/bin/python"
PROJECT_PATH = "/users/visics/kkelchte/code/contrastive-learning"
OUTPUT_PATH = "data/down_stream"

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
NUMEPOCH = 50
# TARGETS = ["cone", "gate", "line"]
TARGETS = ["gate"]
# TARGETS = ["red_line", "line"]
TASKS = ["waypoints", "velocities"]
TEXTURE_DIR = {
    "cone": "data/datasets/dtd",
    "line": "data/datasets/dtd_and_places",
    "red_line": "data/datasets/dtd_and_places",
    "gate": "data/datasets/dtd",
}
CONFIGS = {
    "cone": "default",
    "line": "deep_supervision_fg",
    "red_line": "deep_supervision_fg",
    "gate": "deep_supervision_fg",
}
ENCODERS = {
    "cone": "data/dtd_and_places_augmented/default/cone/best",
    "line": "data/dtd_and_places_augmented/pretrain/deep_supervision_fg/line/best",
    "red_line": "data/dtd_and_places_augmented/pretrain/deep_supervision_fg/red_line/best",
    "gate": "data/dtd_augmented/pretrain/deep_supervision_fg/gate/best",
}

# LEARNING_RATES = [0.001, 0.0001, 0.00001]
LEARNING_RATES = [0.00001]
SUBMIT = True
RM_EXIST = True

print("TARGETS: ", TARGETS)
print("TASKS: ", TASKS)
print("LEARNING_RATES: ", LEARNING_RATES)


def create_condor_job_file(trgt, task, lrate):
    output_dir = f"{OUTPUT_PATH}/{task}/{trgt}/{lrate}"
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} \n")
        jobfile.write(
            f"arguments = {PROJECT_PATH}/run.py --config_file "
            f"{PROJECT_PATH}/configs/{CONFIGS[trgt]}.json "
            f"--learning_rate {lrate} --target {trgt} "
            f"--output_dir {output_dir} --texture_directory {TEXTURE_DIR[trgt]} "
            f"--encoder_ckpt_dir {ENCODERS[trgt]} "
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
    for target in TARGETS:
        for lr in LEARNING_RATES:
            filename = create_condor_job_file(target, task, lr)
            if SUBMIT:
                print(f"submitting {filename}")
                subprocess.call(shlex.split(f"condor_submit {filename}"))
                time.sleep(0.1)
    # wait 20 minutes
    # if SUBMIT and len(CONFIGS) != 1:
    #    time.sleep(10 * 60)

print("finished")
