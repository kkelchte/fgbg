import os
import time
import shutil
import subprocess
import shlex


INTERPRETER_PATH = "/esat/opal/kkelchte/conda/envs/venv/bin/python"
PROJECT_PATH = "/users/visics/kkelchte/code/contrastive-learning"

SPECS = {
    "Universe": "vanilla",
    "Requirements": '(CUDAGlobalMemoryMb >= 3900) && (CUDACapability < 8.6) && (machine != "ruchba.esat.kuleuven.be") && (machine != "dvbrecord.esat.kuleuven.be")  && (machine != "matar.esat.kuleuven.be") && (machine != "jabbah.esat.kuleuven.be")  && (machine != "matar.esat.kuleuven.be") && (machine != "ricotta.esat.kuleuven.be")  && (machine != "amalger.esat.kuleuven.be") && (machine != "amethyst.esat.kuleuven.be")',
    "initial_dir": PROJECT_PATH,
    "priority": 1,
    "RequestCpus": 4,
    "Request_GPUs": 1,
    "RequestMemory": "5 G",
    "RequestDisk": "50 G",
    "Niceuser": "True",
    "+RequestWalltime": int(100 * 3 * 60 * 1.3),
}

TARGETS = ["cone", "gate", "line"]
# TARGETS = ["gate"]
CONFIGS = [
    f"configs/{cf}.json"
    for cf in [
        #"vanilla",
        "default",
        "default_triplet",
        "deep_supervision",
        "deep_supervision_triplet",
        "deep_supervision_blur",
        "deep_supervision_triplet_blur",
    ]
]
LEARNING_RATES = [0.001, 0.0001, 0.00001]
TEXTURE_DIR = "data/places"  # "data/dtd_and_places"  # "data/places" # "data/dtd"

OUTPUT_PATH = f"/users/visics/kkelchte/code/contrastive-learning/data/{os.path.basename(TEXTURE_DIR)}_augmented"

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
    output_dir = f"{OUTPUT_PATH}/{config_tag}/{trgt}/{lrate}"
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} \n")
        jobfile.write(
            f"arguments = {PROJECT_PATH}/run.py --config_file {PROJECT_PATH}/{config} "
            f"--learning_rate {lrate} --target {trgt} "
            f"--output_dir {output_dir} --texture_directory {TEXTURE_DIR}\n"
        )

        for key, value in SPECS.items():
            jobfile.write(f"{key} \t = {value} \n")

        jobfile.write(f"error   = {output_dir}/condor.err\n")
        jobfile.write(f"output  = {output_dir}/condor.out\n")
        jobfile.write(f"log     = {output_dir}/condor.log\n")
        jobfile.write("Queue \n")

    return os.path.join(output_dir, "condor.job")


for conf in CONFIGS:
    for target in TARGETS:
        for lr in LEARNING_RATES:
            filename = create_condor_job_file(target, conf, lr)
            if SUBMIT:
                print(f"submitting {filename}")
                subprocess.call(shlex.split(f"condor_submit {filename}"))
                time.sleep(3)
    # wait 20 minutes
    # if SUBMIT and len(CONFIGS) != 1:
    #    time.sleep(10 * 60)

print("finished")
