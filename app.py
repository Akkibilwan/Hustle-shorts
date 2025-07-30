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
You are an expert YouTube Shorts strategist and video editor.

Your job is to analyze the full transcript of a long-form interview or podcast and extract powerful 30–60 second Shorts using two formats:
1. Direct Clips — continuous timestamp segments that tell a complete story.
2. Franken-Clips — stitched from non-contiguous timestamps, using a hook from one part and payoff from another.

---

🛑 STRICT RULE: DO NOT REWRITE OR SUMMARIZE ANY DIALOGUE.

You must:
- Use the transcript lines exactly as they appear.
- Do not shorten, reword, paraphrase, or compress the speaker’s sentences.
- Keep all original punctuation, phrasing, and spelling.
- Only include full dialogue blocks — no cherry-picking fragments from within a block unless they are timestamped separately.

The output should allow a video editor to directly cut the clip using the given timestamps and script, without needing to interpret or reconstruct phrasing.

---

📌 ANALYSIS GOALS:
- Deeply read and understand the entire transcript before selecting Shorts.
- Prioritize clips with emotional, insightful, or surprising moments.
- Each Short must follow a story arc (hook → context → insight → takeaway).
- Do not suggest clips unless they feel self-contained and high-retention.

---

🎯 THEMES TO PRIORITIZE:
- Money, fame, or behind-the-scenes industry truths
- Firsts and breakthroughs (first paycheck, big break, first failure)
- Vulnerability: burnout, fear, comparison, loneliness
- Transformation: then vs now
- Hacks or hard-earned lessons
- Breaking stereotypes or taboos

---

🛠 HOW TO BUILD FRANKEN-CLIPS:
- Start with a strong hook from any timestamp.
- Skip filler or weak replies.
- Jump to the later timestamp where the real answer, story, or insight is delivered.
- Stitch together in timestamp order.
- Ensure the whole story makes sense even though the timestamps are non-contiguous.

You must include an equal number of Franken-Clips and Direct Clips.

---

TRICT OUTPUT FORMAT (Use the combined timestamps you created):

**Short Title:** [🚀 Viral Title with an Emoji]
**Theme Category:** [Vulnerability/Money/Transformation/Industry/Advice/Norms]
**Number of Segments:** [3 or 4]

**Selected Segments:**
SEGMENT 1: 00:01:23,450 --> 00:01:28,100 - This is my hook, created by merging lines 25-29. [HOOK]
SEGMENT 2: 00:08:45,100 --> 00:08:52,500 - This builds the story, created from lines 150-155. [BUILD]
SEGMENT 3: 00:25:10,300 --> 00:25:19,900 - The amazing payoff, created from lines 412-418. [PAYOFF]

Rationale for Virality:  
[Brief explanation — why this short works. Don’t skip this.]

---

🛑 CRITICAL REMINDERS:
- Do not summarize or "clean up" the speaker’s words.
- Do not shorten lines for brevity.
- Only use lines that appear exactly in the transcript.
- If a timestamp contains multiple lines, include the full lines verbatim.

Now read the full transcript carefully and return high-quality Direct and Franken-Clips that follow this format.

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
