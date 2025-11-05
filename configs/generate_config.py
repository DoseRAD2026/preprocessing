import os
import fnmatch
import yaml

SYNTHRAD2025_BASE = "/data/SynthRAD2025/"

TASK = "1"
ANATOMY = "AB"

OUTPUT_FILE = f"/code/configs/MR_CT_AB_config.yml"

data_dir = os.path.join(SYNTHRAD2025_BASE,f"synthRAD2025_Task{TASK}_Train",f'Task{TASK}', ANATOMY)
cases = fnmatch.filter(os.listdir(data_dir), f"{TASK}{ANATOMY}*")

cases.sort()

for case in cases:
    patient_id = case
    if TASK == "1":
        modality = "MR"
    else:
        modality = "CBCT"
    region = ANATOMY
    ct_path = os.path.join(data_dir, case, f"ct.mha")
    if TASK == "1":
        input_path = os.path.join(data_dir, case, f"mr.mha")
    else:
        input_path = os.path.join(data_dir, case, f"cbct.mha")
    sr_mask = os.path.join(data_dir, case, f"mask.mha")
    center = case[3]
    sr_structures = f"/data_sr/{center}/Task{TASK}/{ANATOMY}/{case}/output/structures/"
    output_dir = f"/data/DoseRAD2026/{modality}_CT/{ANATOMY}/{case}/"
    ts_segmentation = True
    
    data = {
        case:
        {
            "modality": modality,
            "region": region,
            "ct_path": ct_path,
            "input_path": input_path,
            "sr_mask": sr_mask,
            "sr_structures": sr_structures,
            "output_dir": output_dir,
            "ts_segmentation": ts_segmentation
        }
    }

    with open(OUTPUT_FILE, "a") as f:
        yaml.dump(data, f, sort_keys=False)



