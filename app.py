import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google.genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Load API Key ---
# Load from .env file for local development or set as a secret environment variable on Render.
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- LLM, Memory, and Conversation Chain Setup ---
def get_conversation_chain(persona_template):
    """
    Creates and configures a conversation chain with memory.
    """
    if not GOOGLE_API_KEY:
        st.error("🚨 GOOGLE_API_KEY not found. Please set it as an environment variable.")
        st.stop()
    
    # Create the prompt from the selected persona template
    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template=persona_template
    )

    # Initialize the Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",  # ✅ Correct and stable model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0.5 # A slightly higher temperature for more creative responses
    )

    # Initialize memory using Streamlit's session state to persist it across reruns
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    # Create the conversation chain, combining the prompt, LLM, and memory
    conversation_chain = ConversationChain(
        prompt=prompt,
        llm=llm,
        memory=st.session_state.memory
    )
    return conversation_chain

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Universal Academic & Career AI", 
    page_icon="🎓",
    layout="centered"
)

# --- Main UI ---
st.title("🎓 Universal Academic & Career AI")
st.write(
    "Your personal AI guide for academic and career planning. "
    "Choose a persona from the sidebar and start the conversation!"
)

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

# --- Logic to Reset Memory on Persona Change ---
# This ensures a fresh start when the user switches AI roles.
if 'last_agent_choice' not in st.session_state or st.session_state.last_agent_choice != agent_choice:
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.last_agent_choice = agent_choice
    st.session_state.messages = [] # Clear the displayed chat history

# --- Chat History Display ---
# Initialize chat history in session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages from the history on app rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---
if user_query := st.chat_input("Ask your question here..."):
    # 1. Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Select the correct persona template
    template = academic_template if agent_choice == "Expert Academic Advisor" else career_template
    
    # 3. Get the conversation chain
    chain = get_conversation_chain(template)

    # 4. Generate and display the AI's response
    with st.chat_message("assistant"):
        with st.spinner(f"The {agent_choice} is thinking..."):
            # Use the chain's `run` method to get the response
            response = chain.run(user_query)
            st.markdown(response)
    
    # 5. Add the AI's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
