import streamlit as st
from openai import OpenAI

# Initialize client (Hugging Face as backend)
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=st.secrets["api_key"],  # replace with your HF token
)

def qa_analyst_agent(question: str, history: str = "") -> tuple[str, str]:
    
    # System prompt with instructions for clarification and structured output
    system_prompt = """
You are a professional QA Analyst AI. Your role is to assist with software testing tasks, focusing on bugs, test cases, severity, edge cases, and best practices. 

1. If the user's input is vague (e.g., "Login authentication not working") and does not clearly specify an action (e.g., create a bug, generate test cases, suggest causes), respond with: "What would you like me to do? For example, I can create a bug report, generate test cases, or suggest possible causes."
2. If the user specifies an action (e.g., "create a bug"), generate a structured output. For a bug report, include: Summary, Description, Steps to Reproduce, Expected Result, Actual Result, and Elaboration (explaining the issue in simple terms for a QA team).
3. Use conversation history to maintain context and avoid redundant clarification.
4. Keep responses professional, concise, and QA-focused.
5. If the user requests bugs, test cases, severity, edge cases, or best practices, elaborate only on the requested part and provide a summary at the end.

Conversation history: {history}
Question: {question}
Answer:
"""

    # Check for intent (basic keyword matching for simplicity)
    question_lower = question.lower()
    is_vague = not any(keyword in question_lower for keyword in ["bug", "test case", "severity", "edge case", "best practice"])

    # If history indicates a prior vague input and this is a clarification, don't ask again
    if is_vague and "What would you like me to do?" not in history:
        clarification_response = "What would you like me to do? For example, I can create a bug report, generate test cases, or suggest possible causes."
        updated_history = history + f"User: {question}\nAI: {clarification_response}\n"
        return clarification_response, updated_history

    # Build full prompt
    full_prompt = system_prompt.format(history=history, question=question)

    # Call the model
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a helpful QA Analyst AI."},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.4,
    )
    
    # Extract response
    ai_response = response.choices[0].message.content.strip()
    
    # Update history
    updated_history = history + f"User: {question}\nAI: {ai_response}\n"
    
    return ai_response, updated_history

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="QA Analyst AI", page_icon="🕵️", layout="centered")

# Title and description
st.title("🕵️ QA Analyst AI Agent")
st.write("Ask questions and get answers as if you were talking to a QA Analyst.")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of (user_input, ai_response) tuples
if "history" not in st.session_state:
    st.session_state.history = ""  # String history for qa_analyst_agent

# Display conversation history
for user_input, ai_response in st.session_state.conversation:
    with st.container():
        st.markdown(f"**You**: {user_input}")
        st.markdown(ai_response)

# New text input and button for the latest input
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question:", placeholder="e.g., Login authentication not working", key="new_input")
    submit_button = st.form_submit_button("Ask")

# Process input on submit
if submit_button and user_input.strip():
    with st.spinner("Thinking like a QA analyst..."):
        # Call the agent with user input and current history
        answer, updated_history = qa_analyst_agent(user_input, st.session_state.history)
        # Update session state
        st.session_state.conversation.append((user_input, answer))
        st.session_state.history = updated_history
    # Rerun to display new input field
    st.rerun()
elif submit_button and not user_input.strip():
    st.warning("Please enter a question.")