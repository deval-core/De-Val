<div align="center">

# De-Val: The Ultimate Decentralized Evaluation Subnet for LLMs <!-- omit in toc -->
<a href="https://www.de-val.ai">
  <img src="logos/de-val_logo.png" alt="De-Val" width="150"/>
</a>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Miner and Validator Functionality](#miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Roadmap](#roadmap)
- [Running Miners and Validators](#running-miners-and-validators)
- [De-Val Community](#community)
- [License](#license)

---
### Introduction

Welcome to **de_val**, a pioneering decentralized evaluation subnet for Large Language Models (LLMs). Built on the robust BitTensor network, de_val revolutionizes LLM evaluation by promoting a competitive, community-driven approach that enhances model quality, scalability, and innovation. Tackling challenges such as hallucinations, misattributions, relevancy and summary completeness. Our primary focus is evaluation in the context of RAG based scenarios to help businesses answer questions such as:
- How accurate is my LLM given the provided context?
- How comprehensive are our abstractive summaries? Are we missing key details?
- Can we correctly identify key users and match their names to relevant action items? 

Our unique framework enables miners to fine-tune their own models and create custom pre-processing and post-processing pipelines, which are securely submitted and evaluated by validators. This approach addresses critical challenges in LLM outputs, such as hallucinations, misattributions, relevancy, and summary completeness, providing businesses with reliable tools to improve their LLM-based solutions.

### Key Features

🔑 **Decentralized, Competitive Evaluation**

- Contest-based model incentivizes miners to produce high-quality, efficient models.
- Community-driven innovation fosters continuous improvement.

🔍 **Advanced Evaluation Metrics**

- Focus on detecting hallucinations, correcting misattributions, assessing summary completeness, and relevancy.
- Plans to expand metrics to include citation accuracy, date accuracy, and more.

💼 **Business-Focused Solutions**

- Addresses critical market pain points like LLM inaccuracies.
- Provides tools for quality assurance, reducing operational costs, and accelerating time-to-market.

📊 **Detailed Feedback and Analytics**
- Comprehensive scoring from 0 to 1, offering actionable insights for optimizing RAG pipelines.
- Intuitive dashboards for tracking and analyzing performance metrics.

🔄 **Seamless Integration**

- Easy-to-use APIs for integration with existing systems.
- Support for models like the LLaMA family, with plans to expand.

---

## Miner and Validator Functionality

### Miner

- **Model Training and Submission**: Miners train their own models offline using GPU instances and upload the fine-tuned models along with the pipeline code to their Hugging Face repositories.
- **Model Hosting**: Miners run the `contest_miner.py` script to host their models on the network. This script ensures that the models are available for evaluation and integration.
- **Continuous Operation**: Miners keep their miners operational by running the `contest_miner.py` script continuously, allowing their models to participate actively in the network.
- **Model Uniqueness**: Miners are required to imporve upon the base model or eachother's models to ensure uniqueness. They cannot use the same models as other miners without making significant changes.

### Validator

- **Model Retrieval**: Validators download the miners' submitted models and pipeline code from their Hugging Face repositories.
- **Local Testing in Docker**: Validators test the miners' models locally within a Docker container to maintain security and ensure consistent evaluation environments.
- **Task Execution**: Validators run evaluation tasks by sending predefined prompts to the miners' models within the Docker environment to generate responses.
- **Scoring and Evaluation**: Validators assess the performance of each model based on predefined criteria, such as accuracy, relevance, and speed.
- **Logging and Improvement**: Validators log the validation results for auditing and contribute to de_val's wandb project.

## Roadmap

Our key goals for this subnet are:
1. Immediately launch an external API to enable developers an researchers to leverage reference-free evaluations for their research and applications. 
1. Integrate evaluation methods into a synthetic data generation pipeline to ensure that we can generate high-quality and substantive synthetic data for training and benchmarking purposes. 
1. Deploy a new leaderboard based on synthetic data evaluated by models generated on the de_val subnet. Expand trust and transparency in AI benchmarks. 

### Phase 1: Foundation (Q3 2024)
- [x] Launch on testnet
- [x] Develop baseline evaluators
- [x] Launch on Testnet
- [x] Launch website
- [x] Begin marketing for brand awareness and interest
- [x] Integrate contest style validation and winner takes most incentive
- [x] Develop top-down task generation approach (currently only bottoms up generation)

### Phase 2: Expansion (Q4 2024)
- [ ] Launch outward-facing API
- [ ] Achieve competitive baseline evaluation accuracy
- [ ] Collaborations and partnerships with synergistic companies and subnets
- [ ] Introduce next level of evaluation metrics
- [ ] Monetize evaluations through API
- [ ] Develop tools to generate high-quality synthetic data based on our evaluators
- [ ] Launch the de_val leaderboard aimed at providing a constantly changing, reference-free synthetic data leaderboard that is harder to game, but also fully transparent

### Phase 3: Refinement (Q1 2025)
- [ ] Build an open-source database for real-world data submissions
- [ ] Market and sales expansion
- [ ] Expand to additional evaluation metrics such as harmful content detection
- [ ] Explore niche use cases such as regulatory evaluation and privacy compliance
- [ ] Monetize API access to evaluations and proprietary database

## Running Miners and Validators

- [Running a Validator](docs/README_validator.md)

- [Running a Miner](docs/README_miners.md)


## Community

[Discord](https://discord.com/channels/799672011265015819/1272557411948957697)

[WandB](https://wandb.ai/deval-ai/subnet/overview) 

[De-val official website](de-val.ai)


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
