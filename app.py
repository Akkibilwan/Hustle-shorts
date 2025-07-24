import streamlit as st
import io
from openai import OpenAI

# Attempt to import Gemini (Google Generative AI)
gemini_enabled = False
try:
    import google.generativeai as genai
    gemini_enabled = True
except ImportError:
    genai = None

# ----------
# Helper Functions
# ----------

def get_api_key(provider: str) -> str:
    """Retrieve the API key for a given provider from Streamlit secrets."""
    return st.secrets.get(provider, {}).get("api_key", "")

# Initialize OpenAI client
openai_key = get_api_key("openai")
openai_client = OpenAI(api_key=openai_key)

# Initialize Gemini client if available
gemini_key = get_api_key("gemini")
if gemini_enabled:
    genai.configure(api_key=gemini_key)

@st.cache_data(show_spinner=False)
def fetch_models():
    """Fetch available models from both OpenAI and Gemini, with sensible defaults."""
    models = []
    # OpenAI models
    try:
        resp = openai_client.models.list()
        models += [m.id for m in resp.data if m.id.startswith("gpt-")]
    except Exception:
        models += ["gpt-4", "gpt-3.5-turbo"]
    # Gemini models
    if gemini_enabled:
        try:
            gem_models = genai.chat.completions.list_models()
            models += gem_models.get("models", [])
        except Exception:
            models += ["chat-bison-001", "chat-bison-001-pro"]
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for m in models:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    return deduped

# System prompt
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist and editor. Your specialty is converting long-form interviews, podcasts, or conversational transcripts into short-form, high-retention, share-worthy video clips (30–60 seconds).

... [prompt unchanged] ...
"""

# App UI
st.set_page_config(page_title="YouTube Shorts Extractor", layout="wide")
st.title("📽️ Viral YouTube Shorts Extractor")

# File upload
uploaded_file = st.file_uploader("Upload transcript (.srt or .txt)", type=["srt", "txt"])

# Number of Shorts
result_count = st.slider("Number of Shorts to generate", min_value=1, max_value=20, value=5)

# Model selection
available_models = fetch_models()
model = st.selectbox("Choose model", available_models, index=0)

# Generation logic
def generate_shorts(transcript: str, count: int, model_name: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": transcript + f"\n\nPlease generate {count} unique potential shorts in the specified format."}
    ]
    if model_name.startswith("gpt-"):
        resp = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        return resp.choices[0].message.content
    elif gemini_enabled:
        resp = genai.chat.completions.create(
            model=model_name,
            messages=messages
        )
        # Gemini response
        if hasattr(resp, "choices"):
            return resp.choices[0].message.content
        return resp.candidates[0].content
    else:
        st.error(f"Gemini integration not available. Please select an OpenAI model.")
        return None

# Main interaction
if uploaded_file:
    transcript_text = uploaded_file.read().decode("utf-8")
    if st.button("Analyze & Generate Shorts"):
        with st.spinner("Generating viral shorts..."):
            result = generate_shorts(transcript_text, result_count, model)
        if result:
            st.markdown("### Results")
            st.text_area("### Viral Shorts Output", value=result, height=400)

            # Download CSV
            csv_bytes = result.encode("utf-8")
            st.download_button(
                label="Download as CSV",
                data=csv_bytes,
                file_name="shorts_output.csv",
                mime="text/csv"
            )

            # Download RTF as .doc
            rtf_lines = []
            for line in result.split("\n"):
                escaped = line.replace('\\', '\\\\').replace('{', '\\\{').replace('}', '\\\}')
                rtf_lines.append(escaped + "\\par")
            rtf_content = "{\\rtf1\\ansi\n" + "\n".join(rtf_lines) + "\n}"
            st.download_button(
                label="Download as Word (.doc)",
                data=rtf_content.encode("utf-8"),
                file_name="shorts_output.doc",
                mime="application/rtf"
            )
