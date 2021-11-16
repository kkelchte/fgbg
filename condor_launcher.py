import os
import time
import shutil
import subprocess
import shlex


INTERPRETER_PATH = "/esat/opal/kkelchte/conda/envs/venv/bin/python"
PROJECT_PATH = "/users/visics/kkelchte/code/contrastive-learning"

SPECS = {
    "Universe": "vanilla",
    "Requirements": (
        "(CUDAGlobalMemoryMb >= 3900) && (CUDACapability < 8.6) "
        '&& (machine != "vladimir.esat.kuleuven.be") '
        '&& (machine != "kochab.esat.kuleuven.be") '
        '&& (machine != "oculus.esat.kuleuven.be") '
        '&& (machine != "hematite.esat.kuleuven.be") '
        '&& (machine != "bornholm.esat.kuleuven.be") '
        '&& (machine != "egholm.esat.kuleuven.be") '
        '&& (machine != "estragon.esat.kuleuven.be") '
    ),
    "initial_dir": PROJECT_PATH,
    "priority": 1,
    "RequestCpus": 4,
    "Request_GPUs": 1,
    "RequestMemory": "10 G",
    "RequestDisk": "50 G",
    "Niceuser": "True",
    "+RequestWalltime": int(100 * 7 * 60 * 3),
}

CONE_CONFIGS = [
    f"configs/{cf}.json"
    for cf in [
        "deep_supervision_reference_bn",
        "default_fg_bn",
        "default_bn",
    ]
]

RED_LINE_CONFIGS = [
    f"configs/{cf}.json"
    for cf in [
        # "deep_supervision_reference",
        # "deep_supervision_reference_bn",
        "deep_supervision_only_brightness",
    ]
]
GATE_CONFIGS = [
    f"configs/{cf}.json"
    for cf in [
        # "deep_supervision_reference_bn",
        # "deep_supervision_hue_and_brightness_bn",
        "deep_supervision_add_fg_blur_bn",
        "deep_supervision_add_combined_blur_bn",
        "deep_supervision_comb_blur_brightness_hue_bn",
    ]
]

NUM_EPOCHS = {'cone': 50, 'red_line': 50, 'gate': 100}
SUBMIT = True
RM_EXIST = True


def create_condor_job_file(trgt, config, lrate):
    config_tag = os.path.basename(config[:-5])
    output_dir = f"data/{trgt}/{config_tag}"
    if RM_EXIST and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "condor.job"), "w") as jobfile:
        jobfile.write(f"executable     = {INTERPRETER_PATH} \n")
        jobfile.write(
            f"arguments = {PROJECT_PATH}/run.py --config_file {PROJECT_PATH}/{config} "
            f"--learning_rate {lrate} --target {trgt} "
            f"--output_dir {output_dir} --number_of_epochs {NUM_EPOCHS[trgt]}\n"
        )

        for key, value in SPECS.items():
            jobfile.write(f"{key} \t = {value} \n")

        jobfile.write(f"error   = {output_dir}/condor.err\n")
        jobfile.write(f"output  = {output_dir}/condor.out\n")
        jobfile.write(f"log     = {output_dir}/condor.log\n")
        jobfile.write("Queue \n")

    return os.path.join(output_dir, "condor.job")


for conf in RED_LINE_CONFIGS:
    filename = create_condor_job_file("red_line", conf, 0.0001)
    subprocess.call(shlex.split(f"condor_submit {filename}"))

# for conf in GATE_CONFIGS:
#     filename = create_condor_job_file("gate", conf, 0.0001)
#     subprocess.call(shlex.split(f"condor_submit {filename}"))

# for conf in CONE_CONFIGS:
#     filename = create_condor_job_file("cone", conf, 0.001)
#     subprocess.call(shlex.split(f"condor_submit {filename}"))


print("finished")
