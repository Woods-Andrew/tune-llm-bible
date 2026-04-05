import streamlit as st
import boto3
import json
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bible LLM Compare", layout="wide")
st.title("Bible Fine-Tuned LLaMA 3 vs Vanilla")
st.markdown("Enter a prompt and compare the vanilla LLaMA 3 8B response with the Bible fine-tuned version.")

# ── SageMaker runtime client (cached) ───────────────────────────────────────
VANILLA_ENDPOINT = "llama3-8b-vanilla-bible-test"
FINETUNED_ENDPOINT = "llama3-8b-bible-finetuned-v2"


@st.cache_resource
def get_sm_runtime():
    boto_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "default"), region_name="us-east-1")
    return boto_session.client("sagemaker-runtime")


sm_runtime = get_sm_runtime()


def query_endpoint(endpoint_name: str, prompt: str, max_tokens: int = 256) -> str:
    """Invoke a SageMaker endpoint and return generated text."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    }
    response = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read().decode("utf-8"))
    if isinstance(result, list):
        return result[0].get("generated_text", str(result))
    if isinstance(result, dict):
        return result.get("generated_text", str(result))
    return str(result)


# ── Sidebar settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Max new tokens", 64, 1024, 256, step=64)
    st.markdown("---")
    st.caption("Endpoints")
    st.code(VANILLA_ENDPOINT, language=None)
    st.code(FINETUNED_ENDPOINT, language=None)

# ── Prompt input ─────────────────────────────────────────────────────────────
prompt = st.text_area(
    "Prompt",
    placeholder="And the Lord spoke unto his people, saying:",
    height=100,
    key="prompt_input",
)
generate = st.button("Generate", type="primary")

# ── Results ──────────────────────────────────────────────────────────────────
if generate and prompt.strip():
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Vanilla LLaMA 3 8B")
        with st.spinner("Generating..."):
            vanilla_out = query_endpoint(VANILLA_ENDPOINT, prompt, max_tokens)
        st.text_area("Response", vanilla_out, height=400, key="vanilla")

    with col_right:
        st.subheader("Bible Fine-Tuned LLaMA 3 8B")
        with st.spinner("Generating..."):
            finetuned_out = query_endpoint(FINETUNED_ENDPOINT, prompt, max_tokens)
        st.text_area("Response", finetuned_out, height=400, key="finetuned")
