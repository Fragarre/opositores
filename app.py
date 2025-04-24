import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from fpdf import FPDF
from dotenv import load_dotenv
import os

load_dotenv()

# Usuarios autorizados
USERS = {
    os.getenv("USER_ADMIN"): os.getenv("PASS_ADMIN")
}

# Login simple
def login():
    st.title("🔐 Inicio de sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Iniciar sesión")

    if login_button:
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos.")

# Verificar login
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()
    
# Cargar variables de entorno

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Sidebar: configuración del modelo
st.sidebar.title("ℹ️ Información")
st.sidebar.info("Utiliza **sólo el modelo GPT-4o** cuando necesites más precisión. Este modelo usa más recursos y es más caro.")
modelo_seleccionado = st.sidebar.radio("Elige el modelo de lenguaje:", ("gpt-4.1-mini", "gpt-4o"))

# Inicializar modelo
llm = ChatOpenAI(model=modelo_seleccionado, temperature=0)

# Cargar base de vectores FAISS
db = FAISS.load_local("faiss_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un asistente experto en legislación de la Comunidad Valenciana y normativa estatal aplicable.

Tu función principal es ayudar a personas que opositan a cuerpos A1 y A2 de la administración pública valenciana.

Contesta amablemente y con empatía. Intenta dar ánimos al opositor

Usa preferentemente el contexto legal proporcionado a continuación para responder con precisión, citando artículos legales y leyes. 
Evita frases genéricas o sin fundamento legal. Si no encuentras información suficiente en el contexto, puedes complementar con tus conocimientos generales si estás seguro de la respuesta.

Contexto:
{context}

Pregunta:
{question}

Respuesta jurídica fundamentada:
"""
)

# Cadena RAG con prompt legal
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# Herramienta
tools = [
    Tool(
        name="consultar_base_legal",
        func=lambda q: retrieval_chain(q)["result"],
        description="Utiliza esta herramienta para responder preguntas sobre legislación de la Comunidad Valenciana y normativa estatal aplicable."
    )
]

# Memoria y agente
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agente = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

# Función para exportar a PDF
def exportar_a_pdf(pregunta, respuesta):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 10, "Asistente Legal para Opositores - GVA")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Pregunta:\n{pregunta}")
    pdf.ln(3)
    pdf.multi_cell(0, 10, f"Respuesta jurídica:\n{respuesta}")

    pdf_path = "respuesta_legal.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Interfaz principal
st.title("🧑‍⚖️ Asistente para Opositores - GVA")
st.info("Las respuestas pueden tardar. Mientras en la parte superior derecha el Estado sea 'Running', con la figura en movimiento, el sistema esta trabajando. Ten paciencia")

# Sección 1: Consulta legal
st.markdown("### 1️⃣ Consultar legislación")
st.info('Para una nueva consulta, selecciona y borra la anterior')

if "pregunta_input" not in st.session_state:
    st.session_state["pregunta_input"] = ""

st.text_area("Introduce tu pregunta legal o de test:", key="pregunta_input")

col1, col2 = st.columns([1, 0.3])
with col1:
    if st.button("🔍 Consultar"):
        pregunta = st.session_state["pregunta_input"]
        if pregunta:
            with st.spinner("Consultando base legal..."):
                try:
                    resultado = retrieval_chain(pregunta)
                    respuesta = resultado["result"]
                    st.session_state["respuesta"] = respuesta
                    st.session_state["pregunta"] = pregunta
                    st.markdown("### ✅ Respuesta jurídica:")
                    st.write(respuesta)
                except Exception as e:
                    st.error(f"❌ Error durante la consulta: {e}")
        else:
            st.warning("Por favor, introduce una pregunta primero.")

if "respuesta" in st.session_state:
    if st.button("📄 Exportar a PDF"):
        ruta_pdf = exportar_a_pdf(st.session_state["pregunta"], st.session_state["respuesta"])
        with open(ruta_pdf, "rb") as f:
            st.download_button("Descargar PDF", data=f, file_name="respuesta_legal.pdf", mime="application/pdf")

# Sección 2: Pregunta tipo test
st.markdown("### 2️⃣ Practicar con preguntas tipo test")

if "pregunta_test" not in st.session_state:
    st.session_state["pregunta_test"] = None
    st.session_state["opciones"] = []
    st.session_state["opcion_seleccionada"] = None
    st.session_state["explicacion"] = None

if st.button("📝 Generar pregunta tipo test"):
    pregunta_raw = ""
    if "pregunta" in st.session_state and "respuesta" in st.session_state:
        # Intentar generar desde el vectorstore
        resultado = retrieval_chain(st.session_state["pregunta"])
        contexto = resultado["source_documents"]
        if contexto:
            base_contextual = "\n\n".join([doc.page_content for doc in contexto])
            prompt_test_contextual = f"""
Eres un preparador de oposiciones especializado en normativa valenciana.

Basándote exclusivamente en el siguiente contexto legal, genera una única pregunta tipo test con 4 opciones (A, B, C, D). La pregunta debe ser clara y basada en el contenido legal:

{base_contextual}

No des la respuesta correcta aún. Formato:

Pregunta: ...
A) ...
B) ...
C) ...
D) ...
"""
            pregunta_raw = llm.predict(prompt_test_contextual).strip()
        else:
            prompt_fallback = """
Eres un preparador de oposiciones especializado en normativa de la Comunidad Valenciana.

Crea una única pregunta tipo test con 4 opciones (A, B, C, D) sobre legislación autonómica o estatal aplicable a opositores valencianos.

No des la respuesta correcta todavía.
La pregunta debe tener base jurídica real.
Formato:

Pregunta: ...
A) ...
B) ...
C) ...
D) ...
"""
            pregunta_raw = llm.predict(prompt_fallback).strip()
    else:
        # Sin contexto previo
        prompt_general = """
Eres un preparador de oposiciones especializado en normativa de la Comunidad Valenciana.

Crea una única pregunta tipo test con 4 opciones (A, B, C, D) sobre legislación autonómica o estatal aplicable a opositores valencianos.

No des la respuesta correcta todavía.
La pregunta debe tener base jurídica real.
Formato:

Pregunta: ...
A) ...
B) ...
C) ...
D) ...
"""
        pregunta_raw = llm.predict(prompt_general).strip()

    opciones = []
    for letra in ["A)", "B)", "C)", "D)"]:
        idx = pregunta_raw.find(letra)
        if idx != -1:
            opciones.append((letra, pregunta_raw[idx:].split("\n", 1)[0].strip()))

    pregunta_final = pregunta_raw.split("A)")[0].strip()
    st.session_state["pregunta_test"] = pregunta_final
    st.session_state["opciones"] = opciones
    st.session_state["opcion_seleccionada"] = None
    st.session_state["explicacion"] = None

if st.session_state["pregunta_test"]:
    st.markdown(f"**Pregunta de test generada:**\n\n{st.session_state['pregunta_test']}")
    st.radio(
        "Selecciona una opción:",
        options=[texto for _, texto in st.session_state["opciones"]],
        key="respuesta_usuario"
    )

    if st.button("✅ Corregir mi respuesta"):
        opcion = st.session_state["respuesta_usuario"]
        justificacion = llm.predict(f"""
La siguiente es una pregunta tipo test para opositores a la administración pública valenciana:

{st.session_state['pregunta_test']}
{st.session_state['opciones'][0][0]} {st.session_state['opciones'][0][1]}
{st.session_state['opciones'][1][0]} {st.session_state['opciones'][1][1]}
{st.session_state['opciones'][2][0]} {st.session_state['opciones'][2][1]}
{st.session_state['opciones'][3][0]} {st.session_state['opciones'][3][1]}

El opositor ha seleccionado: "{opcion}"

Corrige la pregunta, indica si es correcta o no y justifica la respuesta citando el artículo o ley correspondiente si es posible.
""")
        st.session_state["explicacion"] = justificacion

    if st.session_state["explicacion"]:
        st.markdown("### 🧾 Corrección y justificación:")
        st.write(st.session_state["explicacion"])

# streamlit run streamlit_rag_agente.py