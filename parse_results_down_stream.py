import os
from glob import glob

# data_dir = "data"
data_dir = (
    "/Users/kelchtermans/mount/esat/code/contrastive-learning/data/down_stream"
)
# data_dir = "/Users/kelchtermans/mount/opal/contrastive_learning/dtd_augment"

TARGETS = ["cone", "gate", "line"]
WRITE_WINNING_MODELS = True
WRITE_TABLE = True
TASKS = ["velocities", "waypoints"]

output_dir = f"data/overview_{os.path.basename(data_dir)}"
os.makedirs(output_dir, exist_ok=True)

print("TARGETS: ", TARGETS)
print("TASKS: ", TASKS)
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
overview_results = {t: {c: None for c in TASKS} for t in TARGETS}
winning_lrs = {t: {c: None for c in TASKS} for t in TARGETS}

for target in TARGETS:
    print(f"Parsing target: {target}")
    for tsk in TASKS:
        print(f"Parsing config: {tsk}")
        lr_paths = glob(f"{os.path.join(data_dir, tsk, target)}/*")
        lr_paths = [
            p for p in lr_paths if os.path.exists(os.path.join(p, "results.txt"))
        ]
        values = {
            lrp: get_results_from_txt(os.path.join(lrp, "results.txt"))
            for lrp in lr_paths
        }
        validation_losses = {
            lrp: values[lrp]["validation_mse_loss_avg"] for lrp in lr_paths
        }
        best_lrp = [
            k for k, v in sorted(validation_losses.items(), key=lambda item: item[1])
        ][0]
        overview_results[target][tsk] = values[best_lrp]
        winning_lrs[target][tsk] = best_lrp

if WRITE_TABLE:
    # Print table and store to file:
    overview_file = open(output_dir + "/overview_table.txt", "w")
    for target in ["cone", "gate", "line"]:
        msg = f"{target} && \\\\"
        print(msg)
        overview_file.write("\\hline\n")
        overview_file.write(msg + "\n")
        overview_file.write("\\hline\n")
        for conf in TASKS:
            try:
                msg = f'{os.path.basename(conf).replace("_", " ")} '
                msg += f'&  {overview_results[target][conf]["validation_mse_loss_avg"]} '
                msg += f'(±{overview_results[target][conf]["validation_mse_loss_std"]}) & '
                msg += (
                    f'{overview_results[target][conf]["out-of-distribution_mse_loss_avg"]} '
                )
                msg += (
                    f'(±{overview_results[target][conf]["out-of-distribution_mse_loss_std"]})'
                )
                msg += " \\\\"
                print(msg)
                overview_file.write(msg + "\n")
            except KeyError:
                print(f"Failed to parse {conf}/{target}")
    overview_file.close()

if WRITE_WINNING_MODELS:
    output_file = open(output_dir + "/winning_models.txt", "w")
    for target in TARGETS:
        for conf in TASKS:
            msg = f"{target} - {conf} - {winning_lrs[target][conf]}"
            output_file.write(msg + "\n")
            print(msg)
    output_file.close()

print("finished")
