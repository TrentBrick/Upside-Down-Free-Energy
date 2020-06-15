# Evolutionary Bootstrapping for Active Inference

## Acknowledgements

Thanks to Beren Millidge and Alexander Tschantz for their supervision which made this research possible and successful.

Thanks to [Ha and Schmidhuber, "World Models", 2018]() and the [open source PyTorch implementation](https://github.com/ctallec/world-models) of their code, which provided a solid starting point for the research performed here.

## Instructions for running on a Google Cloud VM:

#### Set up the VM: 
Create a Google Cloud account
Activate Free Credits
Open up Compute Engine
When it finishes setting up your compute engine go to "Quotas" and follow the instructions [here](https://stackoverflow.com/questions/45227064/how-to-request-gpu-quota-increase-in-google-cloud) to request 1 Global GPU.
Wait for approval
Create a new VM under Compute Engine. Choose "From Marketplace" on the left sidebar and search for "Deep Learning" choose the Google Deep Learning VM.
Select a region (east-1d is the cheapest I have seen) and choose the T4 GPU (you can use others but will need to find the appropriate CUDA drivers that I list below yourself.)
Select PyTorch (it has fast.ai too but we dont use this) and ensure it is CUDA 10.1
For installation you can choose 1 CPU but at some point you will want to increase this to 16
Select Ubuntu 16.04 as the OS
Select you want the 3rd party driver software installed (as you will see later we install new drivers so this may be totally unnecessary but I did it and assume you have them installed in later steps)
Add 150 GB of disk space
Launch it.

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
git clone https://github.com/TrentBrick/fem.git
pip3 install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r fem/requirements.txt
sudo apt-get update -y
sudo apt-get install -y xvfb
sudo apt install python-opengl

###Setting up the GPU:::

mkdir ~/Downloads/
mkdir ~/Downloads/nvidia
cd ~/Downloads/nvidia
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
wget http://us.download.nvidia.com/tesla/418.126.02/NVIDIA-Linux-x86_64-418.126.02.run
sudo chmod +x NVIDIA-Linux-x86_64-418.126.02.run
sudo chmod +x cuda_10.1.105_418.39_linux.run
./cuda_10.1.105_418.39_linux.run -extract=~/Downloads/nvidia/

### Uninstall old stuff
sudo apt-get --purge remove nvidia-*
sudo nvidia-uninstall

sudo ./NVIDIA-Linux-x86_64-418.126.02.run --no-opengl-files
sudo ./cuda_10.1.105_418.39_linux.run --no-opengl-libs
### Verify installation
nvidia-smi
```
Your server should now be fully set up to run all of the following experiments!
NB. If you are not using Conda be sure either uninstall it or to call `conda deactivate` every time you SSH in and whenever you start a new tmux terminal.


## Running the code: 

The model is composed of two parts:

  1. A Conditional Variational Auto-Encoder (VAE), this builds a "world model" and the probabilities of visual observations while also producing a useful latent space.
  2. A Mixture-Density Recurrent Network (MDN-RNN), trained to predict the latent encoding of the next frame given past latent encodings, actions and rewards. This uses an LSTM to pass memories over time. It also predicts the future rewards and (optionally) whether or not we are in a terminal state.
  Unlike in World Models we don't learn a policy (controller) and instead use planning with the Cross Entropy Method (CEM) very similar to that used in [1](https://arxiv.org/pdf/2002.12636.pdf), [2](https://arxiv.org/pdf/1811.04551.pdf) and [3](https://arxiv.org/pdf/1805.12114.pdf). Except we don't use an ensemble of models.

Across all training scripts, there are the arguments:
* **--logdir** : The directory in which the models will be stored. If the logdir specified already exists, it loads the old model and continues the training.
* **--noreload** : If you want to override a model in *logdir* instead of reloading it, add this option. NB. This will overwrite the `best.tar` and `checkpoint.tar` models that were previously saved.

### Pretraining
These two models can be first pre-trained separately on simulations using a random policy: 
TODO: NOTE IF THIS USES ACTION REPEATS!

1. Collect simulations using a random policy: 
Before doing this make the directory: `datasets/carracing`
```
python3 data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 16
```

2. Call 
```
python3 trainvae.py --log_dir exp_dir
```

3. Call 
```
python3 trainmdrnn.py --log_dir exp_dir
```

### Joint Training
If you use pretraining, use `cp exp_dir/vae/best.tar exp_dir/joint/vae_best.tar` and 
`cp exp_dir/mdrnn/best.tar exp_dir/joint/mdrnn_best.tar` to have these models loaded in. Be sure to add the `--giving_pretrained` flag when calling the script.

Assuming that you have 16 CPUs:
```
xvfb-run -s "-screen 0 1400x900x24" python3 joint_train.py --log_dir exp_dir --num_workers 16
```
Should you have the `--giving_pretrained` flag on?
There are a number of additional settings at the top of `joint_train.py`.
Fun Fact: `xvfb-run` is necessary for the simulations to run on the headless server. (This is called behind the scenes for `data/generation_script.py` also).

## Evaluating Training

`trainvae.py` and `trainmdrnn.py` write out `logger.json` files and `joint_train.py` outputs a `logger.txt` (I made this as its a big more readable and efficient, also maintains state between training runs). Their learning curves can be plotted using the Jupyter notebooks in `notebooks/plot_MODEL_training.ipynb`.

If you download the trained models, (I would use either `scp` through the Google Cloud SDK command line interface or the VSCode SSH (right click on a file to download it!)) you can run simulations to visualize their performance using `simulated_carracing.py`. In order to see the jointly trained model use:

```
python3 simulated_carracing.py --logdir exp_dir --real_obs --use_planner --test_agent
```
TODO: provide instructions on how to show the dream state and the imagined trajectories the planner creates. 

The following command will allow you to drive the car yourself. Its harder than you would think no?!: 
```
python3 simulated_carracing.py --logdir exp_dir --real_obs
```

## Authors

* **Trenton Bricken** - [trentbrick](https://github.com/trentbrick)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
