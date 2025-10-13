## 📢 Updates
- 2025-10-10: We released our [paper](), Check it out!

## 💾 Installation
Currently only VMWare env is supported. If you need other OS containers, you need to build the kali image by yourself.
### VMware/VirtualBox (Desktop, Laptop, Bare Metal Machine)
Suppose you are operating on a system that has not been virtualized (e.g. your desktop, laptop, bare metal machine), meaning you are not utilizing a virtualized environment like AWS, Azure, or k8s.
If this is the case, proceed with the instructions below. However, if you are on a virtualized platform, please refer to the [Docker](https://github.com/BigComputer-Project/HackWorld?tab=readme-ov-file#docker-server-with-kvm-support-for-the-better) section.

1. First, clone this repository and `cd` into it. Then, install the dependencies listed in `requirements.txt`. It is recommended that you use the latest version of Conda to manage the environment, but you can also choose to manually install the dependencies. Please ensure that the version of Python is >= 3.9.
```bash
# Clone the repository
git clone https://github.com/BigComputer-Project/HackWorld

# Change directory into the cloned repository
cd HackWorld

# Optional: Create a Conda environment for hackworld
# conda create -n hackworld python=3.9
# conda activate hackworld

# Install required dependencies
pip install -r requirements.txt
```

2. Install [VMware Workstation Pro](https://www.vmware.com/products/workstation-pro/workstation-pro-evaluation.html) (for systems with Apple Chips, you should install [VMware Fusion](https://support.broadcom.com/group/ecx/productdownloads?subfamily=VMware+Fusion)) and configure the `vmrun` command.  The installation process can refer to [How to install VMware Worksation Pro](desktop_env/providers/vmware/INSTALL_VMWARE.md). Verify the successful installation by running the following:
```bash
vmrun -T ws list
```
If the installation along with the environment variable set is successful, you will see the message showing the current running virtual machines.


## 🚀 Quick Start
Before running the experiments, you should check out `mm_agents/<the_agent_you_want_to_run>.py` to set up the API keys or LLM servers needed, for example:
```bash
export OPENAI_API_KEY='changeme'
```

Then, simply run the following command to start the evaluation:

```python
python run_ctf_agent.py --config configs/claude_sonnet3_7.yaml --headless --observation_type screenshot
```

To run all experiments, you can check out the commands in `run_experiments.sh`. Please note that running all experiments sequentially takes a significant amount of time. 

### Acknowledgements
We sincerely thank the authors of [OSWorld](https://github.com/xlang-ai/OSWorld) on which HackWorld's GUI controller is built.

## 📄 Citation
If you find this environment useful, please consider citing our work:
```
@misc{HackWorld,
      title={HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities},
      author={Xiaoxue Ren and Penghao Jiang and Kaixin Li and Zhiyong Huang and Xiaoning Du and Jiaojiao Jiang and Zhenchang Xing and Jiamou Sun and Terry Yue Zhuo},
      year={2025},
      eprint={2510.XXXXX},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
