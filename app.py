import streamlit as st
import io
from openai import OpenAI

# ----------
# Helper Functions
# ----------

def get_api_key() -> str:
    """Retrieve the OpenAI API key from Streamlit secrets or environment."""
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    return ""

# Initialize OpenAI client
api_key = get_api_key()
client = OpenAI(api_key=api_key)

# System prompt
epr_system = """
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

**Rationale for Virality:**  
[Why this will perform]

---

Now read the transcript and extract the specified number of unique potential shorts in the above format.
"""

# App UI
st.set_page_config(page_title="YouTube Shorts Extractor", layout="wide")
st.title("📽️ Viral YouTube Shorts Extractor")

# File upload
uploaded_file = st.file_uploader("Upload transcript (.srt or .txt)", type=["srt", "txt"])

# Number of Shorts
result_count = st.slider("Number of Shorts to generate", min_value=1, max_value=20, value=5)

# Model selection
model = st.selectbox("Choose model", ["gpt-4.5", "gpt-4", "gpt-3.5-turbo"], index=0)

def generate_shorts(transcript: str, count: int):
    system = {"role": "system", "content": SYSTEM_PROMPT}
    # Append count requirement
    user_content = transcript + f"\n\nPlease generate {count} unique potential shorts in the specified format."
    user = {"role": "user", "content": user_content}
    response = client.chat.completions.create(
        model=model,
        messages=[system, user],
        temperature=0.7,
        max_tokens=1500
    )
    return response.choices[0].message.content

if uploaded_file:
    transcript_text = uploaded_file.read().decode("utf-8")
    if st.button("Analyze & Generate Shorts"):
        with st.spinner("Generating viral shorts..."):
            result = generate_shorts(transcript_text, result_count)
        # Display
        st.markdown("### Results")
        st.text_area("### Viral Shorts Output", value=result, height=400)

        # Download as CSV
        csv_bytes = result.encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="shorts_output.csv",
            mime="text/csv"
        )

        # Download as Word-compatible RTF
        # Build RTF content to avoid python-docx import issues
        rtf_lines = []
        for line in result.split("\n"):
            escaped = line.replace('\\', '\\\\').replace('{', '\{').replace('}', '\}')
            rtf_lines.append(escaped + "\\par")
        rtf_content = "{\\rtf1\\ansi\n" + "\n".join(rtf_lines) + "\n}"
        rtf_bytes = rtf_content.encode("utf-8")
        st.download_button(
            label="Download as Word (.doc)",
            data=rtf_bytes,
            file_name="shorts_output.doc",
            mime="application/rtf"
        )
