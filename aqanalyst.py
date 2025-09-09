import streamlit as st
from openai import OpenAI

# Initialize client (Hugging Face as backend)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=st.secrests["api_key"],  # replace with your HF token
)

def qa_analyst_agent(question: str):
    prompt = f"""
You are a professional QA Analyst. 
Answer the user's question as a QA Analyst would — focusing on bugs, test cases, severity, edge cases, and best practices.

Question: {question}
Answer:
"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful QA Analyst AI."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="QA Analyst AI", page_icon="🕵️", layout="centered")

st.title("🕵️ QA Analyst AI Agent")
st.write("Ask questions and get answers as if you were talking to a QA Analyst.")

# Input box
user_input = st.text_area("Enter your question:", placeholder="e.g., search link is not working?",height=150)

if st.button("Ask"):
    if user_input.strip():
        with st.spinner("Thinking like a QA analyst..."):
            answer = qa_analyst_agent(user_input)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")



