# Fine-Tune LLaMA 3 8B on the King James Bible

Fine-tune Meta LLaMA 3 8B using Amazon SageMaker JumpStart on the King James Bible, then compare the vanilla and fine-tuned models side by side in a Streamlit app.

## What's Inside

| File | Description |
|------|-------------|
| `tune_llm_bible.ipynb` | End-to-end notebook: download Bible text, prepare JSONL training data, fine-tune with SageMaker JumpStart, deploy, and compare outputs |
| `app.py` | Streamlit web app for side-by-side prompt comparison (vanilla vs fine-tuned) |
| `data/bible_train.jsonl` | Instruction-tuning dataset derived from the King James Bible |
| `test_sagemaker_connection.ipynb` | Quick sanity check for AWS/SageMaker connectivity |

## Architecture

- **Model**: `meta-textgeneration-llama-3-8b`
- **Training**: LoRA fine-tuning on `ml.g5.12xlarge` (4× A10G GPUs)
- **Inference**: Two `ml.g5.xlarge` endpoints (vanilla + fine-tuned)
- **Frontend**: Streamlit app calling SageMaker endpoints via `sagemaker-runtime`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install sagemaker boto3 streamlit
```

Configure your AWS CLI profile with access to SageMaker, then update the `profile_name` in the notebook and set the `AWS_PROFILE` environment variable for the Streamlit app.

## Usage

### Notebook

Run the cells in `tune_llm_bible.ipynb` sequentially:
1. **Setup** — AWS session and role discovery
2. **Download** — Fetch the King James Bible from Project Gutenberg
3. **Prepare** — Create instruction/response JSONL pairs
4. **Fine-tune** — Launch a SageMaker training job (~1–2 hours)
5. **Deploy** — Deploy both vanilla and fine-tuned models
6. **Compare** — Run test prompts through both models
7. **Cleanup** — Delete endpoints to stop charges

### Streamlit App

```bash
export AWS_PROFILE=your_profile
streamlit run app.py
```

Enter a prompt, click **Generate**, and see vanilla vs fine-tuned responses side by side.

## Cost Warning

SageMaker endpoints incur charges while running. Delete them when you're done:
- Via the notebook cleanup cell, or
- `aws sagemaker delete-endpoint --endpoint-name <name>`
