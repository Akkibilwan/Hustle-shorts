
import streamlit as st
import io
from openai import OpenAI
import docx

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

- **Hook (0–3s)**: Shocking number, bold statement, emotional truth, direct question, or stereotype-breaking comment.
- **Context (3–10s)**: Sets up the story with a bit of background.
- **Insight (10–30s)**: The moment of realization, advice, or payoff.
- **Takeaway (30–60s)**: A quote, truth, or punchline that the audience remembers or shares.

---

🔥 THEMES TO PRIORITIZE (in order of virality):
- **Money & Career**: Salary, first paycheck, financial struggles, business realities.
- **Origins & Firsts**: First break, unexpected success, humble beginnings.
- **Emotional Vulnerability**: Self-doubt, rejection, feeling alone, family pressure, burnout.
- **Dark Reality / Industry Secrets**: Behind-the-scenes, hidden struggles, toxic truths.
- **Actionable Advice**: Hacks, mindset shifts, clear life/career lessons.
- **Stereotype-Breaking / Empowerment**: Gender bias, social pressure, comeback moments.
- **Transformation**: “I thought I couldn’t” → “But I did…”

---

🛠 HOW TO CREATE FRANKEN-CLIPS (DO NOT SKIP):
- Identify a strong **hook** (question or opening line).
- Skip filler responses.
- Find the **real answer or payoff** at a later timestamp.
- Combine both in sequence.
- Ensure the stitched clip makes logical and emotional sense.
- Only use this method when the payoff strengthens the hook.

---

📦 OUTPUT FORMAT:

Repeat this format for every Short:

**Potential Short Title:** [Catchy, YouTube-style title with emoji if relevant]  
**Estimated Duration:** [e.g., 45 seconds]  
**Type:** [Direct Clip / Franken-Clip]

**Transcript for Editor:**
| Timestamp | Speaker | Dialogue |
|----------|---------|----------|
| [hh:mm:ss,ms --> hh:mm:ss,ms] | [Name] | [Line] |
| ... | ... | ... |

**Rationale for Virality:**  
[Why this short will perform: strong hook, high relatability, emotional climax, surprise insight, cultural relevance, etc.]

---

⚠️ REMINDERS:
- Don’t include more than 60 seconds. Trim fat aggressively.
- Don’t force clips that don’t complete a narrative.
- Prioritize clarity and emotion over length.
- Choose clips that can stand alone with no additional context.
- You can stitch a franken-clip across any distance in the transcript if it serves the narrative.

---

Now read the entire transcript. Then extract only the most powerful clips — both Direct and Franken — that follow the above format.
"""

# App UI
st.set_page_config(page_title="YouTube Shorts Extractor", layout="wide")
st.title("📽️ Viral YouTube Shorts Extractor")

# File upload
uploaded_file = st.file_uploader("Upload transcript (.srt or .txt)", type=["srt", "txt"])

# Model selection
model = st.selectbox("Choose model", ["gpt-4.5", "gpt-4", "gpt-3.5-turbo"], index=0)

def generate_shorts(transcript: str):
    system = {"role": "system", "content": SYSTEM_PROMPT}
    user = {"role": "user", "content": transcript}
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
            result = generate_shorts(transcript_text)
        # Display
        st.markdown("### Results")
        st.text_area("### Viral Shorts Output", value=result, height=400)

        # Download buttons
        # CSV
        csv_bytes = result.encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="shorts_output.csv",
            mime="text/csv"
        )
        # DOCX
        doc = docx.Document()
        for line in result.split("\n"):
            doc.add_paragraph(line)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        st.download_button(
            label="Download as DOCX",
            data=buf,
            file_name="shorts_output.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

