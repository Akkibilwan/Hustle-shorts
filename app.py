import streamlit as st
import io
from openai import OpenAI, BadRequestError as OpenAIBadRequestError
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as GoogleAPIErrors

# ----------
# Helper Functions
# ----------

def get_openai_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    return ""

def get_google_api_key() -> str:
    """Retrieve the Google AI API key from Streamlit secrets."""
    if "google_ai" in st.secrets and "api_key" in st.secrets["google_ai"]:
        return st.secrets["google_ai"]["api_key"]
    return ""

# --- Model Fetching ---

@st.cache_data(show_spinner=False)
def fetch_openai_models(api_key: str):
    """Fetch available GPT models, fallback to defaults on error."""
    # Fallback list
    default_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    if not api_key:
        return default_models
    try:
        client = OpenAI(api_key=api_key)
        resp = client.models.list()
        models = [m.id for m in resp.data if m.id.startswith("gpt-")]
        # Ensure defaults are present
        for d in default_models:
            if d not in models:
                models.append(d)
        return sorted(list(set(models)))
    except Exception:
        st.error("Could not fetch OpenAI models. Using default list.")
        return default_models

@st.cache_data(show_spinner=False)
def fetch_gemini_models(api_key: str):
    """Fetch available Gemini models, fallback to defaults on error."""
    # Updated fallback list with more modern models
    default_models = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"]
    if not api_key:
        return default_models
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods and 'gemini' in m.name]
        model_ids = [m.split('/')[-1] for m in models]
        if not model_ids:
            return default_models
        # Ensure defaults are present
        for d in default_models:
            if d not in model_ids:
                model_ids.append(d)
        return sorted(list(set(model_ids)))
    except Exception:
        st.error("Could not fetch Gemini models. Using default list.")
        return default_models


# --- System Prompt ---
SYSTEM_PROMPT = """
You are an expert YouTube Shorts strategist and editor. Your specialty is converting long-form interviews, podcasts, or conversational transcripts into short-form, high-retention, share-worthy video clips (30–60 seconds).

Your task is to analyze the provided transcript and identify segments that can be crafted into **viral YouTube Shorts**, using both:
1. **Direct Clips** — continuous timestamps that naturally tell a compelling story.
2. **Franken-Clips** — stitched clips where the hook and payoff occur at different timestamps, but when combined form a powerful narrative.

---

🧠 BEFORE YOU START:
**You must deeply read and understand the entire transcript** before suggesting any Shorts.
- Consider context across the conversation.
- Prioritize Shorts that carry emotional weight, insight, or surprise.
- Do NOT simply return lines based on keyword matches — the short must **make narrative sense**, follow a **viral arc**, and be **audience-relevant**.

---

🎯 VIRAL SHORT STRUCTURE (THE VIRAL ARC):
Every short should ideally follow this structure:

- **Hook (0–3s):** Shocking number, bold statement, emotional truth, direct question, or stereotype-breaking comment.
- **Context (3–10s):** Sets up the story with a bit of background.
- **Insight (10–30s):** The moment of realization, advice, or payoff.
- **Takeaway (30–60s):** A quote, truth, or punchline that the audience remembers or shares.

---

🔥 THEMES TO PRIORITIZE:
- Money & Career
- Origins & Firsts
- Emotional Vulnerability
- Dark Reality / Industry Secrets
- Actionable Advice
- Stereotype-Breaking / Empowerment
- Transformation

---

🛠 HOW TO CREATE FRANKEN-CLIPS:
- Identify a strong **hook**.
- Skip filler.
- Find the **payoff** later.
- Stitch both logically.

---

📦 OUTPUT FORMAT:
Repeat for each Short:
**Potential Short Title:** [Title with emoji]  
**Estimated Duration:** [e.g., 45 seconds]  
**Type:** [Direct Clip / Franken-Clip]

**Transcript for Editor:**
| Timestamp | Speaker | Dialogue |
|----------|---------|----------|
| [hh:mm:ss,ms → hh:mm:ss,ms] | [Name] | [Line] |

**Rationale for Virality:** [Why this will perform]

---

Now read the transcript and extract the specified number of unique potential shorts in the above format.
"""

# --- App UI ---
st.set_page_config(page_title="YouTube Shorts Extractor", layout="wide")
st.title("📽️ Viral YouTube Shorts Extractor")

# --- API Key Initialization ---
openai_api_key = get_openai_api_key()
google_api_key = get_google_api_key()

# --- UI Controls ---
uploaded_file = st.file_uploader("Upload transcript (.srt or .txt)", type=["srt", "txt"])

col1, col2, col3 = st.columns(3)
with col1:
    provider = st.selectbox("Choose AI Provider", ["Google", "OpenAI"])
with col2:
    result_count = st.slider("Number of Shorts to generate", min_value=1, max_value=20, value=5)
with col3:
    if provider == "OpenAI":
        available_models = fetch_openai_models(openai_api_key)
        model = st.selectbox("Choose model", available_models, index=0)
    else: # Google
        available_models = fetch_gemini_models(google_api_key)
        model = st.selectbox("Choose model", available_models, index=0)

# --- Proactive Guidance ---
st.info(
    "**💡 Tip:** If you encounter a 'response was blocked' error, especially with Google's models, "
    "try switching to a different model (like `gemini-1.5-flash`) or changing the AI Provider to OpenAI. "
    "Different models have different safety systems."
)

# --- Generation Logic ---
def generate_shorts(transcript: str, count: int, model_name: str, provider_name: str):
    """Generates shorts using the selected provider's API."""
    user_content = transcript + f"\n\nPlease generate {count} unique potential shorts in the specified format."

    if provider_name == "OpenAI":
        if not openai_api_key:
            st.error("OpenAI API key is not set. Please add it to your Streamlit secrets.")
            return None
        try:
            client = OpenAI(api_key=openai_api_key)
            system = {"role": "system", "content": SYSTEM_PROMPT}
            user = {"role": "user", "content": user_content}
            resp = client.chat.completions.create(
                model=model_name,
                messages=[system, user],
                temperature=0.7,
                max_tokens=3000
            )
            return resp.choices[0].message.content
        except OpenAIBadRequestError as e:
            st.error(f"OpenAI API Error: {e}. The selected model might not be available for your API key.")
            return None
        except Exception as e:
            st.error(f"An unexpected OpenAI API error occurred: {e}")
            return None

    elif provider_name == "Google":
        if not google_api_key:
            st.error("Google AI API key is not set. Please add it to your Streamlit secrets.")
            return None
        try:
            genai.configure(api_key=google_api_key)
            gen_model = genai.GenerativeModel(model_name)
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"
            resp = gen_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=3000
                ),
                safety_settings=safety_settings
            )
            
            if not resp.parts:
                try:
                    finish_reason = resp.candidates[0].finish_reason if resp.candidates else 'UNKNOWN'
                    if finish_reason == 2: # SAFETY
                         st.error("The response was blocked by Google's safety filters. This can sometimes happen even with safe content. **Please try a different model (like Gemini 1.5 Flash) or switch the AI Provider to OpenAI.**")
                    else:
                         st.error(f"The model returned an empty response. Finish Reason: {finish_reason}")
                except IndexError:
                    st.error("The model returned an empty response with no details. This may be due to a safety block or an issue with the prompt.")
                return None

            return resp.text
        except GoogleAPIErrors.InvalidArgument as e:
            st.error(f"Google AI API Error: {e}. Check your prompt or model configuration.")
            return None
        except Exception as e:
            st.error(f"An unexpected Google AI API error occurred: {e}")
            return None
    return None


# --- Main Interaction Logic ---
if uploaded_file:
    transcript_text = uploaded_file.read().decode("utf-8")
    if st.button(f"Analyze & Generate Shorts with {provider}"):
        with st.spinner(f"Generating viral shorts using {provider}'s {model}..."):
            result = generate_shorts(transcript_text, result_count, model, provider)
        
        if result:
            st.markdown("### Results")
            st.text_area("Viral Shorts Output", value=result, height=500)

            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                md_bytes = result.encode("utf-8")
                st.download_button(
                    label="Download as Markdown (.md)",
                    data=md_bytes,
                    file_name="shorts_output.md",
                    mime="text/markdown"
                )
            
            with col_dl2:
                rtf_lines = []
                for line in result.split("\n"):
                    escaped = line.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
                    rtf_lines.append(escaped + "\\par")
                rtf_content = "{\\rtf1\\ansi\n" + "\n".join(rtf_lines) + "\n}"
                st.download_button(
                    label="Download as Word (.doc)",
                    data=rtf_content.encode("utf-8"),
                    file_name="shorts_output.doc",
                    mime="application/rtf"
                )
