
# AWS Bedrock Setup & LLM Support: Step-by-Step Guide

This guide covers how to set up access to AWS Bedrock and includes instructions on integrating models like Anthropic, Mistral, and Cohere with our validation process.

---

## Prerequisites

Before proceeding, make sure you have:
- An active AWS account
- Access to the AWS region where **Bedrock** is available (e.g., **us-east-1**)
- API keys and secrets for using AWS services

---

### Official AWS Guides
- AWS Bedrock [Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-console.html)
- In case you encounter any issues with AWS [permissions](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html#getting-started-bedrock-role)

---

## Step 1: Select Region and Request Bedrock Access

1. **Login to the [AWS Console](https://aws.amazon.com/console/)**.
2. **Select a Region** where AWS Bedrock is supported (e.g., **us-east-1**).
3. Navigate to **Amazon Bedrock** by typing “Bedrock” in the search bar.

   ![](images/aws_access_step_1.png)

4. Click on **"Get Started"**.
   - Once inside the console, navigate to the **bottom** of the side panel.
   - In the **"Bedrock Configuration"** section, select **Model Access**.

   ![](images/aws_access_step_2.png)

5. Select **Modify Model Access**.

   ![](images/aws_access_step_3.png)

6. In the **Bedrock Models** section, request access to models belonging to Anthropic, Mistral, and Cohere.

   ![](images/aws_access_step_4.png)

7. You will be prompted to review the selected models and submit the request. During the request, fill in the required fields:
   - **Company Name**: Enter `de-val`.
   - **Use Case**: Provide a description like “Evaluate the effectiveness of internally developed LLMs.”

   ![](images/aws_access_step_5.png)

---

## Step 2: AWS Access Configuration

1. After receiving access, update your `.env` file to configure the necessary AWS credentials:

    ```bash
    AWS_ACCESS_KEY_ID=your_access_key_here
    AWS_SECRET_ACCESS_KEY=your_secret_key_here
    ```

---

## Step 3: Update Repository & Install Dependencies

1. **Pull the updated repository** by running:

   ```bash
   git pull
   ```

2. **Install the necessary dependencies** using poetry:

   ```bash
   poetry install
   ```

---

## Step 4: Alternative - create dedicated user (more secure)
- Navigate to IAM
- Create New User - name the user `sn-15` 
- Attach `AmazonBedrockFullAccess` policy to the user. 
---

### Key Updates

- **Custom LLM API Integration**: Easily extend the BaseLLM class and integrate with **AWS Bedrock** models (Anthropic, Mistral, Cohere).
- **Task Generator Refactor**: The task generator has been revamped to:
  - Filter models by availability.
  - Ensure models are accessible before task generation.
  - Provide validators full control over which models to use.


