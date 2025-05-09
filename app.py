import streamlit as st
import streamlit.components.v1 as components
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fpdf import FPDF
# from dotenv import load_dotenv
# import os

# load_dotenv()
# Cargar variables de entorno en local
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# --------------------------------------------------------------
# Usuarios autorizados local
# USERS = {
#      os.getenv("USER_ADMIN"): os.getenv("PASS_ADMIN")
#  }
# --------------------------------------------------------------
# Usuarios autorizados deploy
USERS = {
     st.secrets["AUTH"]["USER_ADMIN"]: st.secrets["AUTH"]["PASS_ADMIN"]
 }
# --------------------------------------------------------------

# Login simple
def login():
    st.title("🔐 Inicio de sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    login_button = st.button("Iniciar sesión")

    if login_button:
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("✅ Acceso concedido")
            st.rerun()
        else:
            st.error("❌ Usuario o contraseña incorrectos.")

# Verificar login
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()

# if "memory" not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory(
#         memory_key="chat_history", return_messages=True
#     )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, k=5  # Solo mantiene las últimas k interacciones
    )


# Sidebar: configuración del modelo
st.sidebar.title("ℹ️ Información")
st.sidebar.info("Utiliza **sólo el modelo GPT-4o** cuando necesites más precisión. Este modelo usa más recursos y es más caro.")
modelo_seleccionado = st.sidebar.radio("Elige el modelo de lenguaje:", ("gpt-4.1-mini", "gpt-4o"))

# Inicializar modelo
llm = ChatOpenAI(model=modelo_seleccionado, temperature=0.2)

# Cargar base de vectores FAISS
db = FAISS.load_local("faiss_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
Eres un asistente experto en legislación de la Comunidad Valenciana y normativa estatal aplicable.

Tu función principal es ayudar a personas que opositan a cuerpos A1 y A2 de la administración pública valenciana.

Contesta amablemente.

Usa preferentemente el contexto legal proporcionado a continuación para responder con precisión, citando artículos legales y leyes. 
Evita frases genéricas o sin fundamento legal. Si no encuentras información suficiente en el contexto, puedes complementar con tus conocimientos generales si estás seguro de la respuesta.

Historial de la conversación:
{chat_history}

Contexto:
{context}

Pregunta:
{question}

Respuesta jurídica fundamentada:
"""
)
# memory = ConversationBufferMemory(
#     memory_key="chat_history", return_messages=True
# )

# Cadena RAG con prompt legal
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
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

def borrar_texto():
    st.session_state["pregunta_input"] = ""

def nueva_sesion():
    st.session_state.clear()  # Limpiar toda la sesión
    st.experimental_rerun()    # Reiniciar la app

# Interfaz principal
st.title("🧑‍⚖️ Asistente para Opositores - GVA")
st.info("Las respuestas pueden tardar. Ten paciencia")

# Sección 1: Consulta legal
st.markdown("### 1️⃣ Consultar legislación")
# st.info('Para una nueva consulta, selecciona y borra la anterior')
st.button("🧹 Nueva pregunta", on_click=borrar_texto)
if st.button("🔄 Nueva sesión"):
    nueva_sesion()

if "pregunta_input" not in st.session_state:
    st.session_state["pregunta_input"] = ""

# st.text_area("Introduce tu pregunta legal o de test:", key="pregunta_input")  # ANTES

# Ahora


st.text_area(
    "Introduce tu pregunta legal o de test:",
    key="pregunta_input"
)

col1, col2 = st.columns([1, 0.3])
with col1:
    if st.button("🔍 Consultar"):
        pregunta = st.session_state["pregunta_input"]
        if pregunta:
            with st.spinner("Consultando con contexto..."):
                try:
                    respuesta = retrieval_chain.run(pregunta)
                    st.session_state["respuesta"] = respuesta
                    st.session_state["pregunta"] = pregunta
                    st.markdown("### ✅ Respuesta jurídica:")
                    st.write(respuesta)
                except Exception as e:
                    st.error(f"❌ Error durante la consulta: {e}")
        # Botón para borrar el contenido
    # else:
    #     st.warning("Por favor, introduce una pregunta primero.")

if "respuesta" in st.session_state:
    if st.button("📄 Exportar a PDF"):
        ruta_pdf = exportar_a_pdf(st.session_state["pregunta"], st.session_state["respuesta"])
        with open(ruta_pdf, "rb") as f:
            st.download_button("Descargar PDF", data=f, file_name="respuesta_legal.pdf", mime="application/pdf")

st.markdown("### 2️⃣ Practicar con preguntas tipo test")

st.markdown("Genera preguntas de práctica sobre temas concretos. Puedes pedir hasta 20 preguntas a la vez.")
tema = st.text_input("Indica el tema, ley o disposición sobre la que quieres practicar:")
num_preguntas = st.number_input("¿Cuántas preguntas deseas? (máximo 20)", min_value=1, max_value=20, value=5, step=1)

if "preguntas_test" not in st.session_state:
    st.session_state["preguntas_test"] = []

if st.button("🎯 Generar preguntas de práctica"):
    if not tema.strip():
        st.warning("Por favor, escribe un tema o ley específica.")
    else:
        with st.spinner("Generando preguntas..."):
            st.session_state["preguntas_test"] = []
            for _ in range(num_preguntas):
                prompt_multi = f"""
Eres un preparador de oposiciones experto en legislación valenciana.

Crea UNA pregunta tipo test con 4 opciones (A, B, C, D) sobre el siguiente tema: "{tema}". La pregunta debe:
- Ser de dificultad alta, adecuada para opositores A1/A2.
- Basarse en la interpretación rigurosa o literalidad del texto legal.
- Tener cuatro opciones (A, B, C, D) con distracciones plausibles.

Sigue este formato:
Pregunta: ...
A) ...
B) ...
C) ...
D) ...

No des la respuesta correcta aún.
"""
                pregunta_raw = llm.predict(prompt_multi).strip()
                # Extraer opciones
                opciones = []
                for letra in ["A)", "B)", "C)", "D)"]:
                    idx = pregunta_raw.find(letra)
                    if idx != -1:
                        opciones.append((letra, pregunta_raw[idx:].split("\n", 1)[0].strip()))

                pregunta_texto = pregunta_raw.split("A)")[0].replace("Pregunta:", "").strip()

                st.session_state["preguntas_test"].append({
                    "texto": pregunta_texto,
                    "opciones": opciones,
                    "seleccionada": None,
                    "explicacion": None,
                    "correcta": None
                })

# Mostrar las preguntas
for idx, pregunta in enumerate(st.session_state["preguntas_test"]):
    st.markdown(f"**Pregunta {idx+1}:** {pregunta['texto']}")
    seleccion = st.radio(
        f"Selecciona tu respuesta ({idx+1})", 
        options=[op[1] for op in pregunta["opciones"]],
        key=f"respuesta_{idx}"
    )

    if st.button(f"✅ Corregir {idx+1}"):
        seleccion_usuario = seleccion
        prompt_corrige = f"""
Corrige la siguiente pregunta tipo test, indicando cuál es la respuesta correcta y dando una breve justificación legal clara para opositores:

Pregunta: {pregunta['texto']}
{chr(10).join([opt[1] for opt in pregunta['opciones']])}

Seleccion del usuario: {seleccion_usuario}

Tu formato de respuesta debe ser:
Respuesta correcta: X)
Justificación: ...
"""
        resultado = llm.predict(prompt_corrige).strip()

        correcta = ""
        justificacion = ""
        if "Respuesta correcta:" in resultado:
            partes = resultado.split("Respuesta correcta:")[-1].strip().split("\nJustificación:")
            if len(partes) == 2:
                correcta = partes[0].strip()
                justificacion = partes[1].strip()
        
        if seleccion_usuario.startswith(correcta):
            st.success(f"✅ ¡Correcto! {justificacion}")
        else:
            st.error(f"❌ Incorrecto. La respuesta correcta era {correcta}.")
            st.markdown(f"**Justificación:** {justificacion}")
