import os
import re
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="CBSE Science Tutor",
    page_icon="🔬",
    layout="centered"
)
st.title("🔬 CBSE Science Tutor")
st.caption("Your personal NCERT Science mentor — Class 10")

# ── Language config ───────────────────────────────────────
LANGUAGE_MAP = {
    "English": "English",
    "हिंदी (Hindi)": "Hindi",
    "मराठी (Marathi)": "Marathi"
}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_lang = st.selectbox(
        "🌐 Response Language",
        options=list(LANGUAGE_MAP.keys()),
        index=0
    )
    language = LANGUAGE_MAP[selected_lang]

    st.divider()
    st.header("📝 Quiz")
    st.caption("Test your knowledge on any topic")
    generate_quiz = st.button("🎯 Generate Quiz", use_container_width=True)

    if st.session_state.get("quiz_requested") and "quiz_questions" not in st.session_state:
        quiz_topic = st.text_input(
            "📚 Which topic should I quiz you on?",
            placeholder="e.g. Human Brain, Photosynthesis, Acids and Bases",
            key="quiz_topic_input"
        )
        confirm = st.button("✅ Start Quiz", use_container_width=True)
    else:
        quiz_topic = ""
        confirm = False

    st.divider()
    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        for key in ["quiz_questions", "quiz_requested", "quiz_answers", "quiz_checked"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ── Load components (cached) ──────────────────────────────
@st.cache_resource
def load_components():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "./faiss_db",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    llm = ChatOpenAI(
        base_url="https://api.sarvam.ai/v1",
        api_key=os.environ.get("SARVAM_API_KEY"),
        model="sarvam-m",
        temperature=0.3,
        max_tokens=400
    )
    return retriever, llm

# ── Main QA prompt ────────────────────────────────────────
def build_qa_prompt(language: str) -> PromptTemplate:
    template = f"""You are a strict but caring CBSE Science mentor for Class 10 students.
Your personality: direct, clear, no-nonsense — like a good school teacher who genuinely wants students to learn.
Do NOT be overly friendly or use excessive praise. Be warm but professional.

RESPONSE LANGUAGE: Always respond in {language}.

GREETING RULE — VERY IMPORTANT:
- If the student says hi, hello, hey, thanks, thank you, bye, or any casual greeting or farewell:
  Respond naturally and briefly like a teacher would. Do NOT mention any science topic unprompted.
  Example responses: "Hello! What would you like to study today?" or "You're welcome. Let me know if you have more questions."

ANSWER LENGTH RULES — follow these strictly:
- Simple factual or definition questions (e.g. "What is chlorophyll?"): Answer in 2-3 sentences. Highlight the key term.
- Conceptual or process-based questions (e.g. "Explain photosynthesis"): Answer in detail with steps or structure. Use bullet points if helpful.
- Numerical or formula questions: Show the formula, then solve step by step.

OUT-OF-SYLLABUS RULE:
- If the answer IS in the context below: answer from it directly.
- If the answer is NOT in the context: Start with "[Outside NCERT Syllabus]" then answer briefly from general scientific knowledge. Never make up facts.

NEVER say "I don't know" — either use the textbook context or answer from general knowledge with the label above.

Context from NCERT Textbook:
{{context}}

Student's Question: {{question}}

Answer (in {language}):"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

# ── Quiz prompt ───────────────────────────────────────────
def build_quiz_prompt(topic: str, language: str) -> str:
    return f"""You are a CBSE Science teacher creating a quiz for Class 10 students.

Topic: "{topic}"

Generate exactly 5 multiple choice questions.
Format STRICTLY as shown below — do not deviate:

Q1. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [A or B or C or D]

Q2. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [A or B or C or D]

Q3. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [A or B or C or D]

Q4. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [A or B or C or D]

Q5. [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Answer: [A or B or C or D]

Rules:
- Based on NCERT Class 10 Science only
- One clearly correct answer per question
- Simple, clear language
- Write questions and options in {language}
- Answer line must only contain the letter: A, B, C, or D

Generate the quiz now:"""

# ── Parse quiz text into structured list ──────────────────
def parse_quiz(text: str) -> list:
    questions = []
    blocks = re.split(r'\n(?=Q\d+\.)', text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.match(r'Q\d+\.\s*(.+?)(?=\nA\))', block, re.DOTALL)
        if not q_match:
            continue
        question_text = q_match.group(1).strip()

        options = {}
        for letter in ["A", "B", "C", "D"]:
            opt_match = re.search(rf'{letter}\)\s*(.+?)(?=\n[BCD]\)|\nAnswer:|$)', block, re.DOTALL)
            if opt_match:
                options[letter] = opt_match.group(1).strip()

        ans_match = re.search(r'Answer:\s*([ABCD])', block)
        if not ans_match:
            continue
        answer = ans_match.group(1).strip()

        if len(options) == 4 and answer:
            questions.append({
                "question": question_text,
                "options": options,
                "answer": answer
            })

    return questions

# ── Render interactive quiz ───────────────────────────────
def render_quiz(questions: list):
    if not questions:
        st.warning("Could not parse quiz questions. Please try generating again.")
        return

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_checked" not in st.session_state:
        st.session_state.quiz_checked = {}

    st.markdown("### 📝 Quiz Time!")
    st.caption(f"Answer all {len(questions)} questions and check your score at the end.")

    for i, q in enumerate(questions):
        st.markdown("---")
        st.markdown(f"**Question {i+1}: {q['question']}**")

        option_labels = [f"{k}) {v}" for k, v in q["options"].items()]

        selected = st.radio(
            label=f"q_{i}",
            options=option_labels,
            index=None,
            label_visibility="collapsed",
            key=f"radio_{i}"
        )

        if selected:
            st.session_state.quiz_answers[i] = selected[0]

        col1, col2 = st.columns([1, 4])
        with col1:
            check_clicked = st.button("Check ✅", key=f"check_{i}")

        if check_clicked:
            if not selected:
                st.warning("Please select an option first.")
            else:
                st.session_state.quiz_checked[i] = True

        if st.session_state.quiz_checked.get(i):
            user_ans = st.session_state.quiz_answers.get(i)
            correct_ans = q["answer"]
            correct_text = q["options"][correct_ans]
            if user_ans == correct_ans:
                st.success(f"✅ Correct! The answer is **{correct_ans}) {correct_text}**")
            else:
                st.error(f"❌ Wrong. The correct answer is **{correct_ans}) {correct_text}**")

    # ── Score at the bottom ───────────────────────────────
    checked_count = len(st.session_state.quiz_checked)
    score = sum(
        1 for i, q in enumerate(questions)
        if st.session_state.quiz_checked.get(i) and
        st.session_state.quiz_answers.get(i) == q["answer"]
    )

    if checked_count == len(questions):
        st.markdown("---")
        if score == len(questions):
            st.success(f"🏆 Perfect Score! You scored **{score}/{len(questions)}**. Excellent work!")
        elif score >= 3:
            st.warning(f"👍 Good effort! You scored **{score}/{len(questions)}**. Review the ones you missed.")
        else:
            st.error(f"📖 You scored **{score}/{len(questions)}**. Go back and revise this topic carefully.")

        if st.button("🔁 Retake Quiz", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_checked = {}
            st.rerun()

# ── Build chain with session memory ──────────────────────
def get_chain(retriever, llm, language):
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=4,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    condense_prompt = PromptTemplate.from_template(
        """Given the conversation history and a follow-up question, rephrase the follow-up as a standalone question.
Return ONLY the rephrased question, nothing else.

Chat History:
{chat_history}

Follow-up question: {question}
Standalone question:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_qa_prompt(language)},
        condense_question_prompt=condense_prompt,
        output_key="answer",
        verbose=False
    )
    return chain

# ── Initialise ────────────────────────────────────────────
with st.spinner("Loading your CBSE tutor..."):
    retriever, llm = load_components()

chain = get_chain(retriever, llm, language)

# ── Chat history init ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display previous messages ─────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle Generate Quiz button click ────────────────────
if generate_quiz:
    st.session_state.quiz_requested = True
    for key in ["quiz_questions", "quiz_answers", "quiz_checked"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ── Handle Start Quiz confirmation ───────────────────────
if confirm and quiz_topic.strip():
    with st.spinner("Generating your quiz..."):
        quiz_response = llm.invoke(build_quiz_prompt(quiz_topic.strip(), language))
        raw_quiz = quiz_response.content.strip()
        quiz_text = raw_quiz.split("</think>")[-1].strip() if "</think>" in raw_quiz else raw_quiz
        parsed = parse_quiz(quiz_text)
        st.session_state.quiz_questions = parsed
        st.session_state.quiz_answers = {}
        st.session_state.quiz_checked = {}
        st.session_state.quiz_requested = False
    st.rerun()
elif confirm and not quiz_topic.strip():
    st.sidebar.warning("Please enter a topic first.")

# ── Display interactive quiz ──────────────────────────────
if "quiz_questions" in st.session_state:
    render_quiz(st.session_state.quiz_questions)

# ── Chat input ────────────────────────────────────────────
if question := st.chat_input("Ask a science question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain({"question": question})
            raw_answer = result["answer"]
            answer = raw_answer.split("</think>")[-1].strip() if "</think>" in raw_answer else raw_answer.strip()
            sources = result["source_documents"]

        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources from NCERT"):
                for doc in sources:
                    page = doc.metadata.get("page", "?")
                    source = doc.metadata.get("source", "NCERT")
                    st.caption(f"Page {page} — {source}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
