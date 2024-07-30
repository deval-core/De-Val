<div align="center">

# De-Val: The Ultimate Decentralized Evaluation Subnet for LLMs <!-- omit in toc -->
[![De-Val](path/to/logo.png)](de-val.website)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Bittensor Basics](#bittensor)
  - [Miner](#miner)
  - [Validator](#validator)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Bittensor Basics](#bittensor)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [De-Val Community](#community)
- [License](#license)

---
### Introduction

Welcome to De-Val, the pioneering decentralized evaluation subnet for large language models (LLMs). Built on the robust Bittensor network, De-Val offers unmatched objectivity and reliability in evaluating LLM outputs, tackling challenges such as hallucinations, misattributions, relevancy and summary completeness.

Our unique approach leverages synthetic data for initial training and plans to transition to real data from small to medium-sized businesses (SMBs), ensuring continuous optimization and high standards of AI-generated content.

In our first iteration of the subnet, we will release evaluators focused on:
- Hallucination Detection: Identifying and mitigating false or nonsensical outputs, including:
- Misattribution Correction: Ensuring the accuracy of attributed information.
- Summary Completeness: Evaluating the thoroughness and reliability of generated summaries.
- Relevancy: Binary prediction of whether a response is relevant or not.

## Key Features
🔑 **Unbiased, Decentralized Evaluations**
- Leveraging the power of decentralization to ensure unbiased and reliable assessments.
- Community-driven evaluations for continuous improvement.

🔍 **Advanced Evaluation Metrics**
- Evaluators focused on detecting hallucinations, correcting misattributions, and assessing summary completeness.
- Continuous addition of new evaluators based on community feedback and evolving needs.

💼 **Cost-Effective Solutions for SMBs**
- Flexible pay-per-query API model or subscription, providing affordable access to advanced LLM evaluation tools.
- Tailored solutions for specific business needs.

📊 **Detailed Feedback and Analytics**
- Comprehensive scoring from 0 to 1, offering actionable insights for optimizing RAG pipelines.
- Intuitive dashboards for tracking and analyzing performance metrics.

🔄 **Seamless Integration and Scalability**
- Easy-to-use APIs for integration with existing systems.
- Scalable solutions designed to grow with your business needs.

## Miner and Validator Functionality

### Miner

- Receives evaluation queries from the Validator containing specific information such as context and LLM responses.
- Processes these queries using advanced machine learning models to assess factors like hallucination detection and relevancy.
- Submits detailed evaluation scores back to the Validator for further action and validation.

### Validator

- Collects evaluation results from Miners, including scores and detailed analysis.
- Compares these results against set benchmarks and real-world data, ensuring accuracy and reliability.
- Logs the validation results for auditing and continuous improvement of the evaluation system on the De-Val platform.

## Roadmap

### Phase 1: Foundation (Q3 2024)
- [x] Launch on testnet
- [x] Develop baseline evaluators
- [x] Launch on Testnet
- [ ] Launch website
- [ ] Begin marketing for brand awareness and interest

### Phase 2: Expansion (Q4 2024)
- [ ] Introduce next level of evaluation metrics and validation processes
- [ ] Collaborations and partnerships with synergistic companies and subnets
- [ ] Launch front-end application
- [ ] Build an open-source database for real-world data submissions
- [ ] Achieve competitive baseline evaluation accuracy
- [ ] Monetize evaluations through front-end

### Phase 3: Refinement (Q1 2025)
- [ ] Market and sales expansion
- [ ] Further develop baseline evaluators
- [ ] Expand to additional evaluation metrics such as harmful content detection
- [ ] Explore niche use cases such as regulatory evaluation and privacy compliance
- [ ] Monetize API access to evaluations and proprietary database

## Running Miners and Validators
### Requirements
- Python 3.10+
- Poetry
- CPU only instance
- OPENAI API key

### Bittensor Setup

Follow the bittensor docs [here](https://docs.bittensor.com/getting-started/installation), or paste the following commands into the terminal.
```
sudo apt update && sudo apt install python3-pip

python3 -m pip install bittensor
```

In order to ease management of the scripts and miners running, we recommend using `PM2` and the process manager. In order to istall `PM2`:
```
sudo apt update && sudo apt install jq && sudo apt install npm && sudo npm install pm2 -g && pm2 update
```
For guidance on how to set up your local subtensor click [here](https://docs.bittensor.com/subtensor-nodes/subtensor-node-requirements).

Setting up your wallet can be found [here](https://docs.bittensor.com/getting-started/wallets).

#### Installing poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

Once the above steps are completed log out of your intance and log back in to see `bittensor` and `poetry` installed.

### Validator Setup
We attempted to make everything as easy to run as possible, that is why we went with `poetry` as the package and venv manager.

to set up your validator you will need to follow these steps:
```
git clone https://github.com/deval-core/De-Val.git
cd De-Val

# Adding your OPENAI API key to your .env file
echo "OPENAI_API_KEY=<your_key_here>" > .env

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
btcli s register --subtensor.network <local/test/finney> --netuid <XX> --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

Running Validator:
```
pm2 start neurons/validator.py --name de-val-validator -- \
    --netuid XX
    --subtensor.network <finney/local/test>
    --wallet.name <your coldkey> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode
    --logging.trace # For trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```



### Miner Setup

Just like the validator setup we attempted to make everything as easy to run as possible, that is why we went with `poetry` as the package and venv manager.

to set up your validator you will need to follow these steps:
```
git clone https://github.com/deval-core/De-Val.git
cd De-Val

# Adding your OPENAI API key to your .env file
echo "OPENAI_API_KEY=<your_key_here>" > .env

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
btcli s register --subtensor.network <local/test/finney> --netuid <XX> --wallet.name YOUR_COLDKEY --wallet.hotkey YOUR_HOTKEY
```

Running Validator:
```
pm2 start neurons/miner.py --name de-val-miner -- \
    --netuid XX
    --subtensor.network <finney/local/test>
    --wallet.name <your coldkey> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode
    --logging.trace # For trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
```

## Community

[Discord](link.com)

[X](Link.com)

[De-val official website](xxxx)


## License

De-Val subnet is released under the MIT License.

```
MIT License

Copyright (c) 2023 Opentensor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
