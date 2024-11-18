## Validator set-up 

### Requirements
- Python 3.11
- Poetry
- Docker
- GPU with at least 40GB vram, please see the min_compute.
- NVIDIA drivers  including CUDA
- NVIDIA container-toolkit [please refer to FAQ #3]
- OPENAI API key or AWS Bedrock
- HUGGINGFACE token
- WANDB API key 

### Bittensor Setup

Follow the bittensor docs [here](https://docs.bittensor.com/getting-started/installation), or paste the following commands into the terminal.
```
sudo apt update && sudo apt install python3-pip

python3 -m pip install bittensor
```

In order to ease management of the scripts and miners running, we recommend using `PM2` as your process manager. In order to install `PM2`:
```
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```
For guidance on how to set up your local subtensor click [here](https://docs.bittensor.com/subtensor-nodes/subtensor-node-requirements).

Setting up your wallet can be found [here](https://docs.bittensor.com/getting-started/wallets).

#### Installing poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

Once the above steps are completed log out of your instance and log back in to see `bittensor` and `poetry` installed.

### Installing docker

v1 of de_val runs miner's model in a docker container for security. Therefore docker needs to be enabled.

Quick docker installation:
```
# Get docker 
# Add Docker's official GPG key:
sudo apt-get update -y
sudo apt-get install ca-certificates curl -y
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo docker run hello-world
```
### Choosing models for validation

If you'd like to explicitly define which models to run for validation (e.g., OpenAI, Anthropic, Mistral), please refer to our [Validator Model Selection Guide](docs/Validator_AWS_README.md). This guide includes instructions on how to configure your validator to choose specific models using the `--neuron.model_ids` flag.

### Install packages and dependencies using poetry.

We attempted to make everything as easy to run as possible, that is why we went with `poetry` as the package and venv manager.

To set up your validator you will need to follow these steps:
```
git clone https://github.com/deval-core/De-Val.git
cd De-Val

# Install packages and dependencies using poetry.
poetry install
```
Once that is done, you can activate your venv with `poetry shell`.
```
poetry shell
```
Now that everyhing ready you are ready to launch your validator after registering to the subnet.
example command for registering a validator/miner:
```
btcli s register --subtensor.network <local/test/finney> --netuid <15> --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY

```

### .env set-up

the .env file needs to contain the following tokens:
- HUGGINGFACE token
- WANDB API key 
- OPENAI API key or
- AWS Bedrock

```
WANDB_API_KEY=<your_wandb_key_here>
HUGGINGFACE_TOKEN=<your_hf_key_here>
OPENAI_API_KEY=<your_openai_key_here>
#In case you are using Bedrock:
AWS_ACCESS_KEY_ID=<your_aws_access_key_here>
AWS_SECRET_ACCESS_KEY=<your_aws_seceret_key_here>
```
*Note: the `.env` file should be stored in `~/De-Val/` directory.*

### Build docker container for miner API 

```
docker compose up --build  --timeout 300 miner-api -d
```
This is optional as the code will build the container if not already built. 

Make sure everything in working properly prior to starting your validator using:
```bash
scripts/docker_e2e_test.py 
```
Using `docker_e2e_test.py` you should be able to observe the following:
   - Docker container initializing
   - Task(s) being generated
   - Error logs while attempting to connect to `miner-api` while the model is still loading. 
   - Model being queried successfully.
*Note: You can confirm the model was successfully uploaded and queried using `docker logs miner-api` after running `docker_e2e_test.py`.*
### Running Validator:
```
pm2 start neurons/validator.py --name de-val-validator -- \
    --netuid 15
    --subtensor.network <finney/local/test>
    --wallet.name <your coldkey> 
    --wallet.hotkey <your hotkey> 
    --logging.debug 
    --logging.trace 
    --axon.port <port> 
    --neuron.model_ids 'gpt-4o,gpt-4o-mini,mistral-7b,claude-3.5,command-r-plus'
```

## FAQ 

#### 1. Why is `Poetry` in not being installed correctly with the `curl` command?

There are certain system which do not behave as expected when installing `Poetry` with the `curl` command posted in the guide, alternatively you can install `Poetry` with `pipx` using:
```
if [ "$USER" = "root" ]; then
  apt install pipx
else
  sudo apt install pipx
fi

pipx ensurepath
pipx install poetry
```
#### 2. Should I run my validator with root permissions?


While it's possible to run your validator without root permissions, we **recommend running it with root permissions** to avoid permission-related issues, especially with Docker. Running as root can prevent errors such as the Docker permission error described in FAQ #3. If you choose not to run as root, be prepared to adjust permissions accordingly to ensure proper functionality.

#### 3. What should I do when I see the error:

```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```
This error indicates that the **NVIDIA Container Toolkit** is either not installed or not properly configured on your system. Here's how to resolve it:

**Steps to Resolve:**

1. **Check if NVIDIA Container Toolkit is installed:**

   Open a terminal and run:

   ```bash
   dpkg -l | grep nvidia-container-toolkit
   ```

   - If the command returns nothing, the toolkit is not installed.

2. **Install NVIDIA Container Toolkit:**

   Install the toolkit by running:

   ```bash
   sudo apt install nvidia-container-toolkit -y
   ```

3. **Restart Docker Daemon:**

   After installation, restart the Docker daemon to apply changes:

   ```bash
   sudo systemctl restart docker
   ```

4. **Verify Installation:**

   Test if Docker can access your GPU by running:

   ```bash
   sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   - You should see the output of `nvidia-smi`, indicating that Docker can now use your GPU.

#### 4. What should I do when I see the error **or** running `pm2` without root privileges:

```
Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.47/containers/json?all=1&filters=%7B%22label%22%3A%7B%22com.docker.compose.config-hash%22%3Atrue%2C%22com.docker.compose.project%3Dde-val%22%3Atrue%7D%7D": dial unix /var/run/docker.sock: connect: permission denied
```
This error occurs due to insufficient permissions to access the Docker daemon socket. It typically happens when running Docker commands without root permissions.

**Steps to Resolve:**

1. **Adjust Permissions of Docker Socket:**

   Change the ownership of the Docker socket to your current user:

   ```bash
   sudo chown $USER /var/run/docker.sock
   ```
   **Note:** This change may be reset after a reboot or Docker daemon restart.


#### 5. What should I do if I have further questions or need assistance?

Feel free to reach out to us:

- **Support Channel**: Join our [Discord](https://discord.com/channels/799672011265015819/1272557411948957697) channel for support and discussions.
