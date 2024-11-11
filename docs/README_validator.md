## Validator set-up 

### Requirements
- Python 3.11
- Poetry
- GPU with at least 24GB vram, please see the min_compute.yml for additional details such as CUDA version.
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

### Build docker container for miner API [optional]

```
docker compose up --build  --timeout 300 miner-api
```
This is optional as the code will build the container if not already built. 

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