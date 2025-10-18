import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load API Key ---
# This will load from a .env file for local development.
# On Render, you'll set this as a secret Environment Variable.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LLM (Gemini) ---
# This function creates and configures the AI model
def get_llm():
    if not GOOGLE_API_KEY:
        st.error("🚨 GOOGLE_API_KEY not found. Please set it as an environment variable.")
        st.stop()
        
    return ChatGoogleGenerativeAI(
        # FIX: Changed to the widely supported 'gemini-pro' model
        model="gemini-pro", 
        google_api_key=GOOGLE_API_KEY,
        temperature=0.4
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Universal Academic & Career AI", page_icon="🎓")
st.title("🎓 Universal Academic & Career AI")
st.write("Ask any question, and the AI will answer from its general knowledge.")

st.sidebar.header("Choose an Agent Persona")
agent_choice = st.sidebar.radio(
    "I want the AI to act as a:", 
    ["Expert Academic Advisor", "Seasoned Career Counselor"]
)

user_query = st.text_input("💡 Enter your question here:")

if st.button("Get AI-Powered Answer"):
    if user_query.strip():
        llm = get_llm()

        # --- Use powerful, general-purpose prompts ---
        if agent_choice == "Expert Academic Advisor":
            prompt = f"""
            You are a highly respected Academic Advisor at a leading technology university. Your goal is to provide a comprehensive, structured, and motivational academic roadmap based on the student's request, using your own extensive knowledge of university curricula worldwide.

            **Student's Request:** "{user_query}"

            **Your response MUST be formatted in Markdown and include the following sections:**
            ## 🎓 Academic Roadmap
            ### 📚 Core Subjects & Concepts
            - List critical subjects and concepts. Explain their importance.
            ### 🗓️ Suggested Semester-wise Plan
            - Provide a logical breakdown of subjects per semester for a typical program.
            ### 💡 Key Projects & Practical Skills
            - Suggest 2-3 specific project ideas to build a strong portfolio.
            ### ✨ Final Encouragement
            - End with a motivational closing statement.
            """
        else: # Seasoned Career Counselor
            prompt = f"""
            You are a seasoned Career Counselor specializing in the tech industry. Your task is to create an actionable and insightful career development plan based on the student's request, drawing from your deep understanding of the current job market.

            **Student's Request:** "{user_query}"

            **Your response MUST be formatted in Markdown and include the following sections:**
            ## 💼 Career Development Plan
            ### 🎯 Target Career Paths
            - Identify 2-3 specific and realistic career roles.
            ### 🛠️ Essential Skills & Technologies
            - Create a bulleted list of the top required technical and soft skills.
            ### 🚀 Actionable Next Steps (The First 6 Months)
            - Provide a step-by-step plan (certifications, internships, networking).
            ### 🌟 Professional Branding Tips
            - Give two concrete tips for improving their GitHub or LinkedIn presence.
            """
        
        with st.spinner(f"The {agent_choice} is thinking..."):
            response = llm.invoke(prompt).content
        
        st.subheader("🔎 AI Response:")
        st.markdown(response)
    else:
        st.error("Please enter a question.")
