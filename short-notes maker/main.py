from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Magistral-Small-2506",  # You can change to a better model later
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.set_page_config(page_title="Short Notes Generator", page_icon="📝")
st.title("🧠 AI Short Notes Generator")
st.markdown("Generate concise and customized short notes on any topic using an AI model. Perfect for students, quick revision, or learning new concepts!")

with st.form("notes_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        num_points = st.selectbox("📌 Number of Points", ["3", "5", "10", "15"], index=1)
    with col2:
        tone = st.selectbox("🎯 Tone/Style", ["Precise", "Detailed", "Casual", "Bullet Format Only"])

    format_pref = st.selectbox("📝 Preferred Format", ["Bullet Points", "Numbered List", "Paragraph", "Summary"])
    language = st.selectbox("🌐 Output Language", ["English", "Hindi", "Tamil", "Telugu", "Spanish"])
    topic = st.text_input("🔍 Enter your topic")

    submitted = st.form_submit_button("🚀 Generate Notes")

if submitted:
    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic.")
    else:
        # Prompt
        prompt_template = PromptTemplate(
            input_variables=["topic", "points", "tone", "format", "language"],
            template="""
You are a helpful and intelligent assistant.

Generate short notes on the topic: "{topic}".
- Provide exactly {points} key points.
- Tone/Style: {tone}
- Format the notes as: {format}
- Output language: {language}

Make the output readable and helpful for someone who wants to learn the concept quickly.
Only respond with the notes, no introduction or conclusion.
"""
        )

        prompt = prompt_template.format(
            topic=topic,
            points=num_points,
            tone=tone,
            format=format_pref,
            language=language
        )

        with st.spinner("Generating notes...  [ free api will take time 🥲 ] "):
            try:
                result = model.invoke(prompt)
                st.markdown("### ✨ Your Short Notes")
                st.markdown(result.content)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
