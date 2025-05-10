import streamlit as st
import random
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fpdf import FPDF

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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, k=5  # Solo mantiene las últimas k interacciones
    )

# Sidebar: configuración del modelo
st.sidebar.title("ℹ️ Información")
st.sidebar.info("Utiliza **sólo el modelo GPT-4o** cuando necesites más precisión. Este modelo usa más recursos y es más caro.")
modelo_seleccionado = st.sidebar.radio("Elige el modelo de lenguaje:", ("gpt-4.1-mini", "gpt-4o"))

# Inicializar modelo
llm_consultas = ChatOpenAI(model=modelo_seleccionado, temperature=0.0)
llm_preguntas = ChatOpenAI(model=modelo_seleccionado, temperature=0.2)

# Cargar base de vectores FAISS
db = FAISS.load_local("faiss_db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Prompt personalizado
custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
Eres un asistente experto en legislación de la Comunidad Valenciana y normativa estatal aplicable.

Tu función principal es ayudar a personas que opositan a cuerpos A1 y A2 de la administración pública valenciana.

Contesta de modo profesional como un profesor de Derecho.

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

# Cadena RAG con prompt legal
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_consultas,
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
    st.rerun()    # Reiniciar la app

# Interfaz principal
st.title("🧑‍⚖️ Asistente para Opositores - GVA versión: 0.4")
st.info("Las respuestas pueden tardar. Ten paciencia")

# Sección 1: Consulta legal
st.markdown("### 1️⃣ Consultar legislación")
# st.info('Para una nueva consulta, selecciona y borra la anterior')
st.button("🧹 Nueva pregunta", on_click=borrar_texto)
if st.button("🔄 Nueva sesión"):
    nueva_sesion()

if "pregunta_input" not in st.session_state:
    st.session_state["pregunta_input"] = ""

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

if "respuesta" in st.session_state:
    if st.button("📄 Exportar a PDF"):
        ruta_pdf = exportar_a_pdf(st.session_state["pregunta"], st.session_state["respuesta"])
        with open(ruta_pdf, "rb") as f:
            st.download_button("Descargar PDF", data=f, file_name="respuesta_legal.pdf", mime="application/pdf")
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
            preguntas_generadas = set()
            intentos = 0

            while len(st.session_state["preguntas_test"]) < num_preguntas and intentos < num_preguntas * 4:
                intentos += 1
                semilla = random.randint(1000, 9999)  # ⚠️ Aleatorización del prompt
                contexto_docs = db.similarity_search(tema, k=3)
                contexto = "\n".join([doc.page_content for doc in contexto_docs]) if contexto_docs else ""
                prompt_multi = f"""
Eres un preparador de oposiciones experto en legislación valenciana.

Crea UNA pregunta tipo test con 4 opciones (A, B, C, D) sobre el siguiente tema: "{tema}".

Condiciones:
- Dificultad alta, adecuada para opositores A1/A2.
- Debes estar seguro de que la pregunta se basa en contenidos legales extraidos de la base de datos FAISS.
- Puedes hacer la pregunta como interpretación de un texto legal y un artículo concreto pero no inventes.
- No debes mostrar ni la ley ni el artículo en la pregunta.
- Incluye cuatro opciones plausibles.
- No repitas preguntas previas.

Contexto legal:
{contexto}

Semilla de variación: {semilla}

Formato de salida:
Pregunta: ...
A) ...
B) ...
C) ...
D) ...
"""

                pregunta_raw = llm_preguntas.predict(prompt_multi).strip()
                pregunta_texto = pregunta_raw.split("A)")[0].replace("Pregunta:", "").strip()
                if pregunta_texto in preguntas_generadas:
                    continue  # Evitar repetidas
                preguntas_generadas.add(pregunta_texto)

                opciones = []
                for letra in ["A)", "B)", "C)", "D)"]:
                    idx = pregunta_raw.find(letra)
                    if idx != -1:
                        opciones.append((letra, pregunta_raw[idx:].split("\n", 1)[0].strip()))

                st.session_state["preguntas_test"].append({
                    "texto": pregunta_texto,
                    "opciones": opciones,
                    "seleccionada": None,
                    "explicacion": None,
                    "correcta": None,
                    "corregida": False  # Nuevo campo
                })

# Mostrar preguntas y botones de corrección
for idx, pregunta in enumerate(st.session_state["preguntas_test"]):
    st.markdown(f"**Pregunta {idx+1}:** {pregunta['texto']}")
    seleccion = st.radio(
        f"Selecciona tu respuesta ({idx+1})", 
        options=[op[1] for op in pregunta["opciones"]],
        key=f"respuesta_{idx}"
    )
    if seleccion != pregunta.get("seleccionada"):
        st.session_state["preguntas_test"][idx]["seleccionada"] = seleccion

    if not pregunta.get("corregida"):
        if st.button(f"✅ Corregir {idx+1}", key=f"corregir_{idx}"):
            seleccion_usuario = seleccion
            prompt_corrige = f"""
Corrige la siguiente pregunta tipo test, indicando cuál es la respuesta correcta.

Justifica la respuesta con un breve texto

Utiliza sólo contenido legal de la base de datos FAISS.

Cita la referencia correcta, pero no incluyas la citación textual del contenido, solo ley o disposición y apartado o artículo.

Es muy importante que la citación sea correcta en cuanto a la referencia legal. No inventes.

Termina la respuesta siempre con el texto'Referencia legal: '. Pon a continuación la ley o disposición o reglamento, con el correspondiente artículo o apartado. Siempre aue sea posible incluye el punt del artículo

Pregunta: {pregunta['texto']}
{chr(10).join([opt[1] for opt in pregunta['opciones']])}

Selección del usuario: {seleccion_usuario}

Tu formato de respuesta debe ser:
Respuesta correcta: X)
Justificación: ...
"""
            resultado = llm_preguntas.predict(prompt_corrige).strip()

            correcta = ""
            justificacion = ""
            if "Respuesta correcta:" in resultado:
                partes = resultado.split("Respuesta correcta:")[-1].strip().split("\nJustificación:")
                if len(partes) == 2:
                    correcta = partes[0].strip()
                    justificacion = partes[1].strip()

            # Guardar corrección en el estado
            st.session_state["preguntas_test"][idx]["correcta"] = correcta
            st.session_state["preguntas_test"][idx]["explicacion"] = justificacion
            st.session_state["preguntas_test"][idx]["corregida"] = True

            # Mostrar resultado inmediato
            if seleccion_usuario and correcta:
                if seleccion_usuario.startswith(correcta):
                    st.success(f"✅ ¡Correcto! {justificacion}")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta era {correcta}.")
                    st.markdown(f"**Justificación:** {justificacion}")

    # Si ya fue corregida previamente, mostrar resultado
    elif pregunta.get("corregida"):
        seleccion_usuario = pregunta.get("seleccionada")
        correcta = pregunta.get("correcta")
        justificacion = pregunta.get("explicacion")

        if seleccion_usuario and correcta:
            if seleccion_usuario.startswith(correcta):
                st.success(f"✅ ¡Correcto! {justificacion}")
            else:
                st.error(f"❌ Incorrecto. La respuesta correcta era {correcta}.")
                st.markdown(f"**Justificación:** {justificacion}")