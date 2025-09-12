import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource  # cache model so it loads only once
def load_model():
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",   # ⚠️ use "mps" on Mac M2 for GPU acceleration
        low_cpu_mem_usage=True
    )
    return processor, model

processor, model = load_model()

# -------------------------------
# System Prompt
# -------------------------------
system_prompt = """
You are a professional QA Analyst AI. Your role is to assist with software testing tasks,
focusing on bugs, test cases, severity, edge cases, and best practices.

1. If the user's input is vague (e.g., "Login authentication not working") and does not clearly specify an action 
   (e.g., create a bug, generate test cases, suggest causes), respond with: 
   "What would you like me to do? For example, I can create a bug report, generate test cases, or suggest possible causes."
2. If the user specifies an action (e.g., "create a bug"), generate a structured output. For a bug report, include: 
   Summary, Description, Steps to Reproduce, Expected Result, Actual Result, and Elaboration.
3. Use conversation history to maintain context and avoid redundant clarification.
4. Keep responses professional, concise, and QA-focused.
5. If the user requests bugs, test cases, severity, edge cases, or best practices, 
   elaborate only on the requested part and provide a summary at the end.
"""

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🧑‍💻 QA Analyst AI (Vision + Text)")
st.write("Upload a screenshot and ask a QA-related question.")

# Upload image
uploaded_file = st.file_uploader("Upload screenshot (PNG/JPEG)", type=["png", "jpg", "jpeg"])
question = st.text_input("Enter your question:", placeholder="e.g., Is there any problem with its UI? Create a bug report.")

if uploaded_file and question:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing... please wait ⏳"):
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]}
            ]

            # Convert to prompt
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            # Prepare inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)

            # Generate output
            output_ids = model.generate(**inputs, max_new_tokens=500)
            response = processor.decode(output_ids[0], skip_special_tokens=True)

        # Show result
        st.subheader("💡 AI (QA Analyst) Response:")
        st.write(response)
