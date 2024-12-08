### 🚗 Deep Q Networks using CARLA Driving Simulator

- This repository demonstrates how to run a Reinforcement Learning (RL) experiment with the CARLA Driving Simulator. We implemented the **Deep Q Networks (DQN)** algorithm using **StableBaselines3**.

📚 **Documentation**: [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/en/master/)

---

## 🚀 Instructions

### 1. 🛠️ **Create a Conda Environment**

- Follow the official guide to install Miniconda: [Miniconda Installation](https://docs.anaconda.com/miniconda/)
- Create a new Conda environment:
  
bash
  conda create -n carla python=3.8

  This requires Python 3.8 for proper compatibility with CARLA.

---

### 2. ⚙️ **Activate Your Conda Environment**

In your command line, run:
bash
conda activate [env_name]


---

### 3. 📦 **Install Dependencies**

Install the necessary Python packages using pip:

bash
pip install stable-baselines3
pip install tensorflow
pip install opencv-python


---

### 4. 🏁 **Run CARLA**

Start the CARLA simulator on your terminal at the specified port number:

bash
./CarlaUE4.sh -carla-rpc-port=2000 -norelativemousemode

(Port 2000 is used in this example)

---

### 5. 🖥️ **Open the Files**

- Open train.py and carenv.py in your favorite IDE.
- Activate the Conda environment in your IDE terminal by running:
  
bash
  conda activate [env-name]

  You should see confirmation that the environment is activated.

---

### 6. 🎬 **Run the Training Script**

Execute the following in your terminal to start training your model:
bash
python train.py


---

### 7. 👀 **Watch Your Model Train!**

Enjoy watching your model improve as it learns from the environment!
