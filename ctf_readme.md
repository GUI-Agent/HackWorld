# Introduction
This repo contains code to run GUI agents in a CTF problem-solving enviornment.

# Evaluation

## Preparation
If you want to evaluate claude models using bedrock, please start the litellm server locally to relay the requests.
```bash
cd tools

export AWS_XXX_XXX=your_credentials
export AWS_XXX_YYY=your_credentials

bash launch_litellm_server.sh
```

## Solving the Challenges
We only support using vmware as the VM provider.
```bash
python run_ctf_agent.py --config configs/<model>.yaml --headless
```

## Params
- `--headless` hides the VMWare UI. If you want to inspect it, do not use this param. However, we highly recommend you to use headless mode to avoid:
    - Unexpected VMware dialogs blocking all VM operations
    - VM resolutions changes following the UI window

## Results
Please find the results under the `results` folder. Result summary is stored as `result.json`.

## Caution
- Do not terminate the process when creating VM. To clean up, you should remove `.vmware_vms` file and remove your vm under `vmware_vm_data` completely.
- Use `vmrun list` and `vmrun stop <path> hard` to terminate stuck VMs.
