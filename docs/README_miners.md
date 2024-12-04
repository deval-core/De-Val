## Miner Set-Up

### Requirements

- **Python 3.11** - important for compatibility with PyArmor
- **Poetry**
- **CPU-only instance** for running your miner
- **GPU instance** for training your models offline / testing your models

---

### Important Information About Our Current Approach

1. **Submission to Hugging Face**
   - **Requirements**: Miners must submit their models and pipeline code to Hugging Face.
   - **Repository Size Limit**: The entire Hugging Face repository must not exceed **18GB**. Ensure that your combined model files and pipeline code stay within this limit.
   - **`Coldkey.txt`**: All model repo's need to contain a `.txt` file with the coldkey miners will use to upload their models. Please refer to our base model here for the example of `coldkey.txt`:[deval-core/base-eval/tree/main](https://huggingface.co/deval-core/base-eval/tree/main).
    -**Verification:** We will match this coldkey with your miner UID’s coldkey. Submissions without a matching coldkey will be ignored.
   - **Supported Model Formats**: We currently **only support models in the SafeTensors format**. If you require support for other formats, please let us know—we're open to adding them based on your needs.

2. **Supported Models**
   - **Model Types**: We have tested only on **LLaMA models** so far.
   - **Quantized Models**: At the moment, we **do not support quantized models**. However, we plan to incorporate this feature soon. If there are specific quantizations you’d like us to support, please inform us.

3. **Docker Containers**
   - **Usage**: Docker is utilized exclusively for the **miner API**.

4. **Obfuscation**
   - **Protection**: We allow obfuscation of your code using **PyArmor** to ensure your competitive edge.

5. **Model Duplication Checks**
   - **Base Model Usage**: We will no longer accept submissions that use the base model without any improvements. Only miners who have made improvements(fine-tuning) on the base models will earn rewards. Our base model is available here: [deval-core/base-eval](https://huggingface.co/deval-core/base-eval).
   - **Uniqueness Requirement**: We will check for any duplicated models based on the model's hash submitted on-chain, duplicate models will be checked for commit time on-chain and only the model commited at an earlier block gets evaluated.

6. **Evaluation Process**
   - **Methodology**: We run **30 examples of each task** per iteration.
   - **Grading**: Models are graded based on the average performance across all tasks.

---

## Installation and Setup

### Bittensor Setup

Follow the Bittensor docs [here](https://docs.bittensor.com/getting-started/installation), or paste the following commands into the terminal:

```bash
sudo apt update && sudo apt install python3-pip -y

python3 -m pip install bittensor
```

To ease management of the scripts and miners running, we recommend using **PM2** as your process manager. PM2 is a production-grade process manager for Node.js applications, which helps keep your miner running continuously and restarts it if it crashes.

To install **PM2**:

```bash
sudo apt update && sudo apt install jq npm -y
sudo npm install pm2 -g
pm2 update
```

For guidance on how to set up your local subtensor, click [here](https://docs.bittensor.com/subtensor-nodes/subtensor-node-requirements).

Setting up your wallet can be found [here](https://docs.bittensor.com/getting-started/wallets).

### Installing Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Once the above steps are completed, **log out of your instance and log back in** to see `bittensor` and `poetry` installed. Logging out and back in refreshes your environment variables and PATH, ensuring that `bittensor` and `poetry` are accessible in your terminal.

### Installing Docker

Version 1 of De-Val runs miners' models in a Docker container for security. Therefore, Docker needs to be enabled.

#### Quick Docker installation:

```bash
# Update package lists and install prerequisites
sudo apt-get update -y
sudo apt-get install ca-certificates curl -y

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the Docker repository to Apt sources
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# Test Docker installation
sudo docker run hello-world
```

### .env Setup

The `.env` file needs to contain the following tokens:
- `HUGGINGFACE_TOKEN` - Your Hugging Face token.
- `OPENAI_API_KEY` - (Optional for testing your model)

Example `.env` file:

```env
HUGGINGFACE_TOKEN=<your_hf_key_here>
OPENAI_API_KEY=<your_openai_key_here>
```
### Installing Packages and Dependencies Using Poetry

To simplify the setup process, we use `poetry` for package management and virtual environments.

To set up your miner, follow these steps:

```bash
git clone https://github.com/deval-core/De-Val.git
cd De-Val

# Install packages and dependencies using poetry.
poetry install
```

Once that is done, you can activate your virtual environment with `poetry shell`.

```bash
poetry shell
```

Now that everything is ready, you are ready to launch your miner after registering to the subnet.

Example command for registering a validator/miner:

```bash
btcli s register --subtensor.network <local/test/finney> --netuid <15> --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

---

## Main Miner Tasks

As a miner, you are required to complete the following two primary tasks:

1. **Submit Your Model and Pipeline Code to Hugging Face**
   - **Description**: Upload your fine-tuned model along with the accompanying pipeline code to Hugging Face for evaluation and integration.
   - **Procedure**:
     - **Script Location**: Use the `submit.py` script located in the `De-Val/scripts/` directory.
     - **Execution**: Run the script by following the detailed usage instructions provided below to ensure a successful submission.

2. **Run and Maintain Your Miner with `contest_miner.py`**
   - **Description**: After submitting your model, you must run your miner to participate actively in the contest. This involves keeping the miner operational, which is separate from both the training and submission processes.
   - **Procedure**:
     - **Script Configuration**:
       - Open the `contest_miner.py` script located in `neurons/miners`.
       - You would need to manually modify the variables `repo_id` and `model_id`.
       - Input your **Repository ID** and **Model ID** corresponding to your Hugging Face submission.
     - **Running the Miner**:
       - Execute the `contest_miner.py` script to start the miner.
       - Ensure that this script remains running continuously to maintain active participation.
     - **Important Notes**:
       - **Continuous Operation**: The miner must be kept running at all times during the contest to ensure proper evaluation and scoring. Running your miner can be done on any hardware and does not require a GPU as you would for training the model.
       - **Separation of Concerns**: Running the miner is a distinct task from training and submitting your model.
     - **Resources**:
       - **Examples**: Refer to the examples provided in the **`neurons/`** folder of our repository to guide your setup and configuration.

---

## Code Format and Requirements

We currently only support code in the form of **Hugging Face pipelines**. Please adhere to the following requirements:

### Data Format

Your pipeline must output data in the following format:

```python
{
    'score_completion': float,
    'mistakes_completion': list[str]
}
```

- **score_completion**: A float representing the score of the completion.
- **mistakes_completion**: A list of strings describing any mistakes found.

### Pipeline Code Requirements

- **Directory Structure**: The pipeline code must be within a folder called **`model/`**.
- **Main File**: The core file with your pipeline code must be named **`pipeline.py`**.
- **Pipeline Class**: The pipeline class must be called **`DeValPipeline`**.
- **Example**: Follow the example provided at **`neurons/miners/model/`**.

### Testing Your Pipeline

To test your model and see if everything is uploaded as intended you will need to run run on the same architecture required by validators to ensure your pipeline works correctly before submitting. You can find out how to set up the validator using the [Validator's guide](docs/README_validator.md).

 We have create a set of scripts to help you test your pipeline's success and whether your model will work using the script:
```bash
scripts/docker_e2e_test.py 
```
```bash
scripts/e2e_test.py 
```
```bash
scripts/generate_task_examples.py
```
---

## Obfuscation and Submission - `submit.py`

We provide a script to perform obfuscation and submit your model and pipeline code to your Hugging Face repository:

```bash
scripts/submit.py
```

### Example Usage

An example command to use the submission script is:

```bash
poetry run python3 scripts/submit.py \
    --model_dir ../eval_llm \
    --repo_id your-username/your-model \
    --pipeline_dir neurons/miners/model \
    --upload_pipeline \
    --upload_model \
    --wallet_name your_coldkey_name \
    --hotkey_name your_hotkey_name \
    --test # Only use if you are submitting your model to testnet 
    --hf_token # can be skipped if you have it in your .env file
```
*note: Do not obfuscate your code unless you are actually uploading a pipeline.*

- **Replace** `your-username/your-model` with your actual Hugging Face repository ID.
- **Flags**:
  - `--model_dir`: Path to your model directory.
  - `--repo_id`: Your Hugging Face repository ID.
  - `--pipeline_dir`: Path to your pipeline code.
  - `--wallet_name`: Your coldkey name.  
  - `--hotkey_name`: Your hotkey name.  
  - `--test`: Add this flag for testnet submissions only. 
  - `--upload_pipeline`: Include this flag to upload the pipeline code.
  - `--upload_model`: Include this flag to upload the model.
  - `--hf_token`: Your Hugging Face token. (You can skip this if you set your HF token in `.env` with `HUGGINGFACE_TOKEN='Your_HF_Token'`)
- Both `--upload-pipeline` and `--upload-model` are optional, but you need to choose at least one.
---

### Running your miner - `contest_miner.py`

Once you have trained your model, customized your pipeline, and submitted it to Hugging Face, you need to run your miner.

**Note:** Running your miner can be done on any hardware and does not require a GPU as you would for training the model.

- Modify lines **variables**: `synapse.repo_id` and `synapse.model_id` of `contest_miner.py` to point at your own model that you submitted using `submit.py`.
- Register a UID as stated above.
- Run your miner.

Example usage of `contest_miner.py`:

```bash
pm2 start neurons/contest_miner.py --name de-val-miner -- \
    --netuid 15 \
    --subtensor.network <finney/local/test> \
    --wallet.name <your coldkey> \
    --wallet.hotkey <your hotkey> \
    --logging.debug \
    --logging.trace \
    --axon.port <port>
```

**Note:** The `--` is used to separate `pm2` options from the script's options.

---

## Additional Notes

- **Package Dependencies**:
  - If your model requires additional packages not included at runtime, **let us know**, and we will review them. For security reasons, all packages must be approved.

- **PyArmor Considerations**:
  - **PyArmor** can be finicky with platforms and may only run when platforms match. Ensure that you obfuscate your code on the same platform where it will run, as PyArmor is sensitive to platform differences. We specify the platform in the obfuscation pipeline to mitigate this issue.

- **Model Uniqueness**:
  - We do not allow miners to use the same models as one another.
  - We verify uniqueness through **hashes**, but you may use other miners' models to further train your own.
  - We will match this coldkey with your miner UID’s coldkey. Submissions without a matching coldkey will be ignored.
- **Evaluation Frequency**:
  - We do not evaluate all UIDs every time.
  - We only evaluate the **top 20 models by incentive** or if your model had a commit in the last **48 hours**.

## FAQ

#### 1. How do I test my model and pipeline using your scripts?

To conduct tests using our scripts, you need to set up a **validator instance** on a GPU with sufficient VRAM and an **x86 architecture**. Please refer to the **validator's README** for detailed setup instructions and requirements.

#### 2. What Python version is required for running PyArmor?

You must use **Python 3.11** (ideally 3.11.8) when running PyArmor for code obfuscation. Using a different Python version may lead to compatibility issues during the obfuscation process or when validators execute your code.

#### 3. Can I change the `platform` flag in `obfuscate.py` for PyArmor?

No, **do not change the `platform` flag** in the `obfuscate.py` script used for PyArmor. Altering this flag can cause your obfuscated code to be incompatible with the validators' execution environment, leading to errors during evaluation.

#### 4. How do you check for duplicate models?

We check for duplicate models based on the **commit date of the entire repository** on Hugging Face. Be cautious when making updates to your Hugging Face repository:

- **Avoid unnecessary commits**: Only commit significant changes to prevent altering the commit date without reason.
- **Model Uniqueness**: Ensure that your model has substantial differences from others, except for the base LLaMA 3 8B model, which everyone can use as a starting point.

#### 5. What should I do if my model requires additional packages?

If your model needs additional packages not currently included at runtime:

- **Contact Us**: Let us know the packages you require.
- **Review Process**: We will review and approve necessary packages for security reasons.
- **Approval Required**: Do not include unapproved packages in your submission.

#### 6. How can I avoid issues with code obfuscation?

- **Follow Guidelines**: Adhere strictly to the provided instructions for obfuscation.
- **Do Not Modify Scripts**: Avoid changing any settings in the `obfuscate.py` script unless instructed.
- **Testing**: Use the provided testing scripts to verify your obfuscated code works as expected before submission.

#### 7. What should I do if I have further questions or need assistance?

Feel free to reach out to us:

- **Support Channel**: Join our [Discord](https://discord.com/channels/799672011265015819/1272557411948957697) channel for support and discussions.


