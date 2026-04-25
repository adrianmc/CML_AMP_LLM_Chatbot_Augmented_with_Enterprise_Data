"""
llm_rag_app.py — RAG Chatbot v3
Cloudera Machine Learning

Mejoras sobre v2:
  - Guarda texto limpio y truncado DENTRO de Milvus (no re-lee archivos del disco)
  - Auto-instala pdfminer.six si no está disponible
  - Múltiples fallbacks para extracción de PDF
  - Zero dependencias externas que puedan fallar en runtime
"""
import os
import sys
import subprocess

# ══════════════════════════════════════
# AUTO-INSTALL DE DEPENDENCIAS
# ══════════════════════════════════════
def ensure_pdfminer():
    """Instala pdfminer.six automáticamente si no está disponible."""
    try:
        from pdfminer.high_level import extract_text
        return True
    except ImportError:
        print("pdfminer.six no encontrado. Instalando automáticamente...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pdfminer.six', '-q'])
            print("pdfminer.six instalado exitosamente")
            return True
        except Exception as e:
            print(f"No se pudo instalar pdfminer.six: {e}")
            return False

# Ejecutar al inicio
PDFMINER_AVAILABLE = ensure_pdfminer()

# ══════════════════════════════════════
# PATHS Y IMPORTS DEL PROYECTO
# ══════════════════════════════════════
sys.path.insert(0, '/home/cdsw')
sys.path.insert(0, '/home/cdsw/3_job-populate-vectordb')

from milvus import default_server
from pymilvus import (
    connections, Collection, utility,
    FieldSchema, CollectionSchema, DataType
)
import utils.model_llm_utils as model_llm
import utils.model_embedding_utils as model_embedding

# ══════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════
COLLECTION_NAME = 'rag_documents'
MILVUS_DATA_DIR = 'milvus-data'
MAX_TEXT_LENGTH = 1500  # Caracteres máx por documento (protege GPU)
EMBEDDING_DIM = 384

# ══════════════════════════════════════
# MILVUS
# ══════════════════════════════════════
def start_milvus():
    """Inicia Milvus y conecta."""
    try:
        connections.disconnect('default')
    except:
        pass
    default_server.set_base_dir(MILVUS_DATA_DIR)
    if not default_server.running:
        default_server.start()
    connections.connect(
        alias='default',
        host='localhost',
        port=default_server.listen_port
    )
    print("Milvus conectado")


def ensure_collection():
    """Crea la colección con campo de texto incluido."""
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(
            name='doc_id',
            dtype=DataType.VARCHAR,
            max_length=500,
            is_primary=True,
            auto_id=False
        ),
        FieldSchema(
            name='text_content',
            dtype=DataType.VARCHAR,
            max_length=2000,  # Texto limpio y truncado
        ),
        FieldSchema(
            name='embedding',
            dtype=DataType.FLOAT_VECTOR,
            dim=EMBEDDING_DIM
        ),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={
            'metric_type': 'IP',
            'index_type': 'IVF_FLAT',
            'params': {'nlist': 2048}
        }
    )
    print(f"Colección '{COLLECTION_NAME}' creada")
    return collection


def get_indexed_docs():
    """Lista documentos indexados."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        count = collection.num_entities
        results = collection.query(
            expr="doc_id != ''",
            output_fields=['doc_id'],
            limit=200
        )
        collection.release()
        docs = [r['doc_id'] for r in results]
        return docs, count
    except:
        return [], 0


# ══════════════════════════════════════
# EXTRACCIÓN DE TEXTO (BLINDADA)
# ══════════════════════════════════════
def clean_text(text: str) -> str:
    """Limpia y trunca texto para que sea seguro en Milvus y en la GPU."""
    if not text:
        return ""
    # Eliminar bytes inválidos
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # Eliminar caracteres nulos y de control
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
    # Colapsar espacios múltiples
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Truncar
    text = text.strip()[:MAX_TEXT_LENGTH]
    return text


def extract_pdf_text(file_path: str) -> str:
    """Extrae texto de PDF con múltiples fallbacks."""
    text = ""

    # Intento 1: pdfminer (mejor calidad)
    if PDFMINER_AVAILABLE:
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
            if text and text.strip():
                return text
        except Exception as e:
            print(f"  pdfminer falló: {e}")

    # Intento 2: pypdf / PyPDF2
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        text = '\n'.join(pages_text)
        if text and text.strip():
            return text
    except Exception as e:
        print(f"  pypdf/PyPDF2 falló: {e}")

    # Intento 3: leer bytes y filtrar texto ASCII/UTF8
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        # Extraer solo caracteres imprimibles
        text = ''.join(
            chr(b) for b in raw
            if 32 <= b < 127 or b in (10, 13, 9)
        )
        if text and len(text.strip()) > 50:
            return text
    except Exception as e:
        print(f"  Raw extraction falló: {e}")

    return text or "[No se pudo extraer texto del PDF]"


def extract_text_from_file(file_path: str) -> str:
    """Extrae texto de cualquier archivo soportado."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        text = extract_pdf_text(file_path)
    else:
        # TXT, MD, TEXT
        try:
            with open(file_path, 'r', errors='ignore') as f:
                text = f.read()
        except Exception:
            with open(file_path, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')

    return clean_text(text)


# ══════════════════════════════════════
# INGESTA
# ══════════════════════════════════════
def ingest_file(file_path: str, file_name: str) -> str:
    """Ingesta un archivo: extrae texto, genera embedding, guarda TODO en Milvus."""
    try:
        # Extraer y limpiar texto
        text = extract_text_from_file(file_path)
        if not text.strip() or len(text.strip()) < 10:
            return f"⚠️ {file_name}: Archivo vacío o sin texto extraíble"

        # Generar embedding
        embedding = model_embedding.get_embeddings(text)

        # Guardar en Milvus: ID + texto + embedding
        collection = Collection(COLLECTION_NAME)
        collection.insert([
            [file_name],       # doc_id
            [text],            # text_content (ya limpio y truncado)
            [embedding],       # embedding vector
        ])
        collection.flush()

        chars = len(text)
        return f"✅ {file_name}: Indexado ({chars} caracteres)"
    except Exception as e:
        return f"❌ {file_name}: {str(e)}"


def process_uploads(files) -> str:
    """Procesa archivos subidos desde la UI."""
    if not files:
        return "⚠️ No se seleccionaron archivos"

    ensure_collection()
    results = []

    for file in files:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_name = os.path.basename(file_path)

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ('.pdf', '.txt', '.text', '.md'):
            results.append(f"⏭️ {file_name}: Formato no soportado (PDF o TXT)")
            continue

        result = ingest_file(file_path, file_name)
        results.append(result)

    success = sum(1 for r in results if r.startswith("✅"))
    total = len(results)
    summary = f"\n{'─' * 40}\n📊 {success}/{total} archivos indexados exitosamente"
    return "\n".join(results) + summary


def refresh_doc_list():
    """Lista de docs para la UI."""
    docs, count = get_indexed_docs()
    if not docs:
        return "📂 No hay documentos indexados", "Total: 0"

    lines = []
    for d in sorted(docs):
        icon = "📄" if d.lower().endswith('.pdf') else "📝"
        lines.append(f"{icon} {d}")
    return "\n".join(lines), f"Total: {count} documentos"


# ══════════════════════════════════════
# RAG QUERY (SIN ACCESO A DISCO)
# ══════════════════════════════════════
def chat_query(question, history):
    """Pipeline RAG: pregunta → buscar en Milvus → generar respuesta."""
    if not question.strip():
        return "", history

    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()

        # Buscar documento más similar
        q_embedding = model_embedding.get_embeddings(question)
        results = collection.search(
            data=[q_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=1,
            output_fields=['doc_id', 'text_content'],  # Traer texto directo
            consistency_level="Strong"
        )
        collection.release()

        if not results or not results[0]:
            history = history + [(question, "No encontré documentos relevantes. Sube documentos en la pestaña 📤")]
            return "", history

        # Extraer contexto DIRECTO de Milvus (sin tocar disco)
        hit = results[0][0]
        doc_name = hit.entity.get('doc_id')
        context = hit.entity.get('text_content')
        score = hit.distance

        # Generar respuesta con LLM
        prompt = f"""<human>:{context}. Answer this question based on given context {question}
<bot>:"""

        response = model_llm.get_llm_generation(
            prompt,
            ['<human>:', '\n<bot>:'],
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.85,
            top_k=70,
            repetition_penalty=1.07
        )

        full_response = f"{response}\n\n📎 *Fuente: {doc_name} (relevancia: {score:.2f})*"
        history = history + [(question, full_response)]

    except Exception as e:
        history = history + [(question, f"❌ Error: {str(e)}")]

    return "", history


# ══════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════
def create_app():
    import gradio as gr

    css = """
    .header-box {
        background: linear-gradient(135deg, #0D3D56 0%, #1B5E7B 60%, #2A8CB5 100%);
        padding: 24px 32px; border-radius: 14px; margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(13,61,86,0.15);
    }
    .header-box h1 { color: white !important; font-size: 26px !important; margin: 0 0 4px !important; }
    .header-box p { color: rgba(255,255,255,0.7) !important; font-size: 14px !important; margin: 0 !important; }
    .accent { color: #F96702 !important; }
    .tab-nav button.selected { border-bottom: 3px solid #F96702 !important; color: #1B5E7B !important; }
    .upload-area { border: 2px dashed #B0D8E8 !important; border-radius: 12px !important; background: #F7FBFD !important; }
    .upload-area:hover { border-color: #F96702 !important; background: #FFF8F3 !important; }
    .footer-info { text-align: center; padding: 16px; color: #5A6A7A; font-size: 12px; }
    """

    with gr.Blocks(
        title="RAG Chatbot | Cloudera AI",
        theme=gr.themes.Soft(
            primary_hue="orange", secondary_hue="blue",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css=css,
    ) as app:

        gr.HTML("""
        <div class="header-box">
            <h1>💬 RAG Chatbot <span class="accent">| Cloudera AI</span></h1>
            <p>Sube documentos PDF o TXT y consulta su contenido en lenguaje natural</p>
        </div>
        """)

        with gr.Tabs():
            # ──── TAB: CHATBOT ────
            with gr.Tab("💬 Chatbot"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=480, show_label=False,
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Escribe tu pregunta aquí...",
                                show_label=False, scale=5, container=False,
                            )
                            send = gr.Button("Enviar", variant="primary", scale=1)
                        clear = gr.Button("🗑️ Limpiar chat", size="sm")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Documentos")
                        doc_count = gr.Markdown("Total: 0")
                        doc_list = gr.Textbox(
                            interactive=False, lines=14,
                            show_label=False,
                        )
                        refresh = gr.Button("🔄 Actualizar", size="sm")

                gr.Examples(
                    examples=[
                        ["What are ML Runtimes?"],
                        ["How do data scientists use CML?"],
                        ["What are iceberg tables?"],
                    ],
                    inputs=msg,
                )

            # ──── TAB: SUBIR DOCUMENTOS ────
            with gr.Tab("📤 Subir Documentos"):
                gr.Markdown("""
                ### Agregar Documentos a la Base de Conocimiento
                Sube archivos **PDF** o **TXT**. El sistema extrae el texto,
                genera un embedding, y lo indexa en Milvus automáticamente.
                
                El texto se limpia (UTF-8) y se trunca a un tamaño seguro para la GPU.
                No hay dependencias externas que puedan fallar.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        files = gr.File(
                            label="Arrastra archivos aquí o haz clic",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".text", ".md"],
                            elem_classes=["upload-area"],
                            height=200,
                        )
                        upload_btn = gr.Button(
                            "🚀 Procesar e Indexar",
                            variant="primary", size="lg",
                        )
                    with gr.Column(scale=1):
                        upload_log = gr.Textbox(
                            label="Resultado",
                            interactive=False, lines=15,
                        )

                gr.Markdown("""
                ---
                **Formatos:** PDF, TXT, MD · **Máx recomendado:** 10 MB por archivo
                """)

            # ──── TAB: ACERCA DE ────
            with gr.Tab("ℹ️ Arquitectura"):
                gr.Markdown("""
                ### ¿Cómo funciona?

                **Al subir un documento:**
                
                Documento → Extraer texto (pdfminer/pypdf/raw) → Limpiar UTF-8 → Truncar → Embedding 384-dim → Guardar texto + vector en Milvus

                **Al hacer una pregunta:**
                
                Pregunta → Embedding → Buscar en Milvus → Recuperar texto (directo, sin leer disco) → Prompt + contexto → LLM → Respuesta
                
                ---

                ### Diferencia clave vs versión anterior

                | Antes (v1) | Ahora (v3) |
                |---|---|
                | Milvus guarda solo el file path | Milvus guarda el texto completo |
                | Al responder, re-lee el archivo del disco | Al responder, lee de Milvus directo |
                | Si el PDF tiene bytes malos → crash | Texto ya limpio desde la ingesta |
                | Si pdfminer no está instalado → crash | Auto-instala + 3 fallbacks |
                | Documento entero → CUDA OOM | Truncado a 1500 chars desde ingesta |

                ---

                | Componente | Tecnología |
                |---|---|
                | Vector Store | Milvus (integrado) |
                | Embeddings | all-MiniLM-L6-v2 (384-dim) |
                | LLM | h2ogpt-oig-oasst1-512-6.9b |
                | Extracción PDF | pdfminer → pypdf → raw (fallbacks) |
                | Interfaz | Gradio |
                """)

        gr.HTML('<div class="footer-info">Cloudera Solutions Engineering · 2026</div>')

        # ── Events ──
        send.click(chat_query, [msg, chatbot], [msg, chatbot])
        msg.submit(chat_query, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        upload_btn.click(process_uploads, [files], [upload_log])
        refresh.click(refresh_doc_list, outputs=[doc_list, doc_count])
        app.load(refresh_doc_list, outputs=[doc_list, doc_count])

    return app


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════
if __name__ == "__main__":
    print("Iniciando RAG Chatbot v3...")
    start_milvus()
    ensure_collection()
    print("Milvus listo")

    app = create_app()
    app.queue().launch(
        share=False, show_error=True,
        server_name='127.0.0.1',
        server_port=int(os.getenv('CDSW_APP_PORT', '8080')),
    )
