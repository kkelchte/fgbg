from genericpath import exists
import os
import shutil
from glob import glob

# data_dir = "data"
data_dir = (
    "/Users/kelchtermans/mount/esat/code/contrastive-learning/data/places_augmented"
)
# data_dir = "/Users/kelchtermans/mount/opal/contrastive_learning/dtd_augment"

TARGETS = ["cone", "gate", "line"]
COPY_REAL_IMGS = True
WRITE_WINNING_MODELS = True
WRITE_TABLE = True

# "vanilla",
CONFIGS = [
    "default",
    "default_triplet",
    "deep_supervision",
    "deep_supervision_triplet",
    "deep_supervision_blur",
    "deep_supervision_triplet_blur",
]
output_dir = f"data/overview_{os.path.basename(data_dir)}"
os.makedirs(output_dir, exist_ok=True)

print("TARGETS: ", TARGETS)
print("CONFIGS: ", CONFIGS)
print("OUTPUTDIR: ", output_dir)

def get_results_from_txt(filename) -> dict:
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        return {ln.split(":")[0]: float(ln.split(":")[1]) for ln in lines if ":" in ln}
    except FileNotFoundError:
        return {}


# Get an overview of the quantitative results
# and keep the paths of the best learning rates
overview_results = {t: {c: None for c in CONFIGS} for t in TARGETS}
winning_lrs = {t: {c: None for c in CONFIGS} for t in TARGETS}

for target in TARGETS:
    print(f"Parsing target: {target}")
    for conf in CONFIGS:
        print(f"Parsing config: {conf}")
        lr_paths = glob(f"{os.path.join(data_dir, conf, target)}/*")
        lr_paths = [
            p for p in lr_paths if os.path.exists(os.path.join(p, "results.txt"))
        ]
        values = {
            lrp: get_results_from_txt(os.path.join(lrp, "results.txt"))
            for lrp in lr_paths
        }
        validation_losses = {
            lrp: values[lrp]["validation_bce_loss_avg"] for lrp in lr_paths
        }
        best_lrp = [
            k for k, v in sorted(validation_losses.items(), key=lambda item: item[1])
        ][0]
        overview_results[target][conf] = values[best_lrp]
        winning_lrs[target][conf] = best_lrp

if WRITE_TABLE:
    # Print table and store to file:
    overview_file = open(output_dir + "/overview_table.txt", "w")
    for target in ["cone", "gate", "line"]:
        for conf in CONFIGS:
            try:
                msg = f'{os.path.basename(conf).replace("_", " ")} '
                msg += f'&  {overview_results[target][conf]["validation_iou_avg"]} '
                msg += f'(±{overview_results[target][conf]["validation_iou_std"]}) & '
                msg += (
                    f'{overview_results[target][conf]["out-of-distribution_ious_avg"]} '
                )
                msg += f'(±{overview_results[target][conf]["out-of-distribution_ious_std"]})'
                msg += " \\\\"
                print(msg)
                overview_file.write(msg + "\n")
            except KeyError:
                print(f"Failed to parse {conf}/{target}")
    overview_file.close()

# Copy winning real images for quantitative results
if COPY_REAL_IMGS:
    for target in TARGETS:
        for conf in CONFIGS:
            try:
                shutil.copyfile(
                    winning_lrs[target][conf] + "/imgs/real_0.jpg",
                    f"{output_dir}/{target}_{os.path.basename(conf)}.jpg",
                )
            except FileNotFoundError:
                print(f"Failed to copy from {winning_lrs[target][conf]}")

if WRITE_WINNING_MODELS:
    output_file = open(output_dir + "/winning_models.txt", "w")
    for target in TARGETS:
        for conf in CONFIGS:
            msg = f"{target} - {conf} - {winning_lrs[target][conf]}"
            output_file.write(msg)
            print(msg)

print("finished")
