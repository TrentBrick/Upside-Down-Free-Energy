# Evolutionary Bootstrapping for Upside Down Active Inference

An exploration into how to better apply the Free Energy Principle to Reinforcement Learning and convergences with the recent rise of "upside down" supervised reinforcement learning. 

See `TrentonBricken_FEP_Upside_Down_RL_Paper.pdf` for a draft write up of the project that also has a summary of the Free Energy Principle. 

I was unable to get the inverse RL agent to ultimately learn useful hierarchical goals and haven't worked on this project for two years so I am open sourcing the codebase. 

This codebase is a more advanced and unweildy version of https://github.com/TrentBrick/RewardConditionedUDRL that was used to replicate two upside down RL papers. 

## Relevant Scripts: 

* `train.py` - has almost all of the relevant configuration settings for the code. Also starts either ray tune (for hyperparam optimization) or a single model (for debugging). Able to switch between different model and learning types in a modular fashion
* `bash_train.sh` - uses GNU parallel to run multiple seeds of a model
* `lighting-trainer.py` - meat of the code. Uses pytorch lightning for training
* `control/agent.py` - runs rollouts of the environment and processes their rewards
* `envs/gym_params.py` - provides environment specific parameters
* `exp_dir/` - contains all experiments separated by: environment_name/experiment_name/seed/logged_versions
* `models/upsd_model.py` - contains the [Schmidhuber](https://arxiv.org/pdf/1912.13465.pdf) and [Levine](https://arxiv.org/abs/1912.02877) upside down models.
* `models/advantage_model.py` - model to learn the advantage of actions as in the Levine paper

## Running the code: 

To run a single model of the lunar-lander call:

```
python trainer.py --gamename lunarlander \                                                  
--exp_name levine_reward_norm_weights \
--num_workers 1 --no_reload --seed 25
```

Environments that are currently supported are lunarlander and lunarlander-sparse. Where the sparse version gives all of the rewards at the very end.

To run multiple seeds call `bash bash_train.sh` changing the trainer.py settings and experiment name as is desired.

To run Ray hyperparameter tuning, uncomment all of the `ray.tune()` functions for desired hyperparamters to search over and set `use_tune=True`.

## Evaluating Training

All training results along with important metrics are saved out to Tensorboard. To view them call: 

`tensorboard --logdir fem/exp_dir/*ENVIRONMENT_NAME*/*EXPERIMENT_NAME*`

To visualize the performance of a trained model, locate the model's checkpoint which will be under: `exp_dir/*ENVIRONMENT_NAME*/*EXPERIMENT_NAME*/*SEED*/epoch=*VALUE*.ckpt` and put this inside `load_name = join(game_dir, 'epoch=1940_v0.ckpt')` in trainer.py then call the code with with correct experiment name and `--eval 1` flag.

## Instructions for running on a Google Cloud VM:

#### Set up the VM: 
* Create a Google Cloud account
* Activate Free Credits
* Open up Compute Engine
* When it finishes setting up your compute engine go to "Quotas" and follow the instructions [here](https://stackoverflow.com/questions/45227064/how-to-request-gpu-quota-increase-in-google-cloud) to request 1 Global GPU.
* Wait for approval
* Create a new VM under Compute Engine. Choose "From Marketplace" on the left sidebar and search for "Deep Learning" choose the Google Deep Learning VM.
* Select a region (east-1d is the cheapest I have seen) and choose the T4 GPU (you can use others but will need to find the appropriate CUDA drivers that I list below yourself.)
* Select PyTorch (it has fast.ai too but we dont use this) and ensure it is CUDA 10.1
* For installation you can choose 1 CPU but at some point you will want to increase this to 16
* Select Ubuntu 16.04 as the OS
* Select you want the 3rd party driver software installed (as you will see later we install new drivers so this may be totally unnecessary but I did it and assume you have them installed in later steps)
* Add 150 GB of disk space
* Launch it.

The next two subheaders are if you want to be able to SSH into the server from your IDE (instructions provided for VS Code) (I recommend this!). But if you want to use the SSH button via Google Cloud thats fine too.

#### Connect Static IP 
In the top search bar look up "External IP" select it. Create a new static IP address. Attach it to your new VM.

(You may need to turn off and back on your VM for this to take effect.)

#### IDE SSH
For VSCode I use the installed plugin "Remote Explorer". 
My ssh keys are in ~/.ssh so I do `cat ~/.ssh/id_rsa.pub` and copy and paste this into the SSH section of Google Cloud (Search for SSH). 

Then with my server on I get its external IP address and in VSCode remote explorer call: 
`ssh -i ~/.ssh/id_rsa SSH_KEY_USERNAME@SERVER_IP_ADDRESS`
Before following the instructions.
One thing that first caught me up is that you need to give the ssh prefix not the the specific .pub file!

#### Installing Dependencies 
With access to your server sorted, you now need to install a few dependencies:

Note - this VM comes with Conda preinstalled along with Python 2.7 and 3.5 outside of Conda. We will not be using Conda so either uninstall it or every time you log on ensure you deactivate it first! (If you can get this all working with Conda then good for you but I had problems with it early on in this project and so decided to ignore it.)

Note2 - The CUDA drivers are installed with open-gl which we do not want in order to be able to run the CarRacing Gym environment headlessly on the server. As a result we need to reinstall these drivers using the `--no-opengl-files` flag.

If you are running this on a local computer, all you need is: 

```
pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r fem/requirements.txt
```

If you are running this on a server and your GPU expects OpenGL, then run all of the below. I run it in chunks because the CUDA installers have interactive pages.
```
conda init
exit
### reopen terminal. Conda was there but couldn't be deactivated till you do this!
conda deactivate
## deleta conda (if you want, else will need to call conda deactivate) every time
sudo rm -rf ../../opt/conda/
git clone https://github.com/TrentBrick/fem.git
sudo apt-get update

pip3 install --upgrade pip
pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r fem/requirements.txt
sudo apt-get update -y
sudo apt-get install -y xvfb

###Setting up the GPU:::

mkdir ~/Downloads/
mkdir ~/Downloads/nvidia
cd ~/Downloads/nvidia
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
wget http://us.download.nvidia.com/tesla/418.126.02/NVIDIA-Linux-x86_64-418.126.02.run
sudo chmod +x NVIDIA-Linux-x86_64-418.126.02.run
sudo chmod +x cuda_10.1.105_418.39_linux.run
./cuda_10.1.105_418.39_linux.run -extract=~/Downloads/nvidia/

### Uninstall old stuff. Choose default options. You may get some warning messages. Which is fine. 
sudo apt-get --purge remove nvidia-*
sudo nvidia-uninstall

sudo ./NVIDIA-Linux-x86_64-418.126.02.run --no-opengl-files
sudo ./cuda_10.1.105_418.39_linux.run --no-opengl-libs
### Verify installation
nvidia-smi

# install opengl.
sudo apt-get install python-opengl
```
Your server should now be fully set up to run all of the following experiments! Please don't post Issues on installation as I won't be able to provide any further support and have already provided a lot more than most other ML code reproductions/support!
NB. If you are not using Conda be sure either uninstall it or to call `conda deactivate` every time you SSH in and whenever you start a new tmux terminal.

## Acknowledgements

Thanks to [Beren Millidge](https://berenmillidge.github.io/aboutme/) and [Alexander Tschantz](https://alec-tschantz.github.io/) for their supervision which made this research possible and successful.

Thanks to [Ha and Schmidhuber, "World Models", 2018]() and the [open source PyTorch implementation](https://github.com/ctallec/world-models) of their code, which provided a solid starting point for the research performed here. Thanks also to the opensource implementation of Upside Down Reinforcement Learning: https://github.com/jscriptcoder/Upside-Down-Reinforcement-Learning which provided an initial test base. Also to [Reward Conditioned Policies](https://arxiv.org/pdf/1912.13465.pdf) and [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877) for initial research and results (I just wish both of these papers shared their code...).

## Authors

* **Trenton Bricken** - [trentbrick](https://github.com/trentbrick)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
