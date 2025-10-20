import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google.genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Load API Key ---
# Load from .env file for local development. On Render, set this as a secret environment variable.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Page Configuration ---
st.set_page_config(
    page_title="Universal Academic & Career AI",
    page_icon="🎓",
    layout="centered"
)

# --- Main Application ---
st.title("🎓 Universal Academic & Career AI")

# --- API Key Check ---
# Stop the app if the API key is not found. This is a critical check.
if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found. Please set it as an environment variable in your deployment settings.")
    st.stop()

# --- LLM, Memory, and Conversation Chain Setup ---
def get_conversation_chain(persona_template):
    """Creates and configures a conversation chain with memory."""

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=persona_template
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Correct and stable model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5
    )

    # Initialize memory using Streamlit's session state to persist it across reruns
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    conversation_chain = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=st.session_state.memory
    )
    return conversation_chain

# --- UI Sidebar ---
st.sidebar.header("Choose an Agent Persona")
agent_choice = st.sidebar.radio(
    "I want the AI to act as a:",
    ["Expert Academic Advisor", "Seasoned Career Counselor"],
    key="agent_choice",
    help="Select the AI's role. Changing this will start a new conversation."
)

# --- Persona Prompt Templates ---
academic_template = """
You are a highly respected Academic Advisor at a top-tier technology university.
Your goal is to provide a comprehensive, structured, and motivational academic roadmap.
You must maintain this persona throughout the entire conversation.

Current conversation:
{history}

Student: {input}
Academic Advisor:
"""

career_template = """
You are a seasoned Career Counselor specializing in the technology industry.
Your task is to create actionable and insightful career development plans.
You must maintain this persona throughout the entire conversation.

Current conversation:
{history}

Student: {input}
Career Counselor:
"""

# --- State Management (Resetting Memory on Persona Change) ---
if 'last_agent_choice' not in st.session_state or st.session_state.last_agent_choice != agent_choice:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.last_agent_choice = agent_choice
    # Add a welcome message when the conversation starts or resets
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am your {agent_choice}. How can I assist you today?"}
    ]

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---
if user_query := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    template = academic_template if agent_choice == "Expert Academic Advisor" else career_template
    chain = get_conversation_chain(template)

    with st.chat_message("assistant"):
        with st.spinner(f"The {agent_choice} is thinking..."):
            response = chain.run(user_query)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
