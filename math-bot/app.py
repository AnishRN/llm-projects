import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# ─────────────────────────────────────────────────────────────
# 🎨 Streamlit Page Setup
st.set_page_config(
    page_title="Text to Math Solver & Wiki Assistant",
    page_icon="🧮",
    layout="centered"
)

st.title("🧠 Text → Math Solver & Data Assistant")
st.markdown("Ask math-related or general knowledge questions, and get detailed answers using Google Gemma 2!")

# ─────────────────────────────────────────────────────────────
# 🔐 API Key Input
groq_api_key = st.sidebar.text_input(label="🔑 Groq API Key", type="password")
if not groq_api_key:
    st.info("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# 🤖 Initialize LLM
try:
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
except Exception as e:
    st.error("❌ Failed to initialize LLM. Check your API key or model name.")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────────────────────
# 🔧 Tool: Wikipedia
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Searches Wikipedia for general knowledge and topic info."
)

# 🔧 Tool: Calculator (MathChain)
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Solves math expressions and provides calculations."
)

# 🔧 Tool: Reasoning Chain (LLMChain)
reasoning_prompt = """
You're an intelligent assistant tasked with solving the user's math or logic problem.
Please think through the problem step-by-step and provide a detailed explanation, formatted in bullet points.

Question: {question}
Answer:
"""

reasoning_template = PromptTemplate(
    input_variables=["question"],
    template=reasoning_prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=reasoning_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="Handles logical reasoning and multi-step problems."
)

# ─────────────────────────────────────────────────────────────
# 🤝 Initialize the Agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# ─────────────────────────────────────────────────────────────
# 💬 Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi there! I can solve your math problems and help with research. Ask me anything!"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ─────────────────────────────────────────────────────────────
# 🧮 Input Question
question = st.text_area(
    label="📥 Enter your question:",
    value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?",
    height=150
)

# ─────────────────────────────────────────────────────────────
# 🚀 Generate Answer
if st.button("🔍 Find My Answer"):
    if question.strip():
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("🧠 Thinking..."):
            try:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = assistant_agent.run(question, callbacks=[st_cb])
                st.session_state["messages"].append({"role": "assistant", "content": response})

                st.chat_message("assistant").write(response)
                st.success("✅ Response generated!")
            except Exception as e:
                st.error("Something went wrong while generating the answer.")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a valid question before clicking.")
