"""
llm_rag_app.py — RAG Chatbot v4
Cloudera Machine Learning
Compatible con Gradio 6.0+

Features:
  - Barra de progreso por archivo durante indexación
  - Animaciones CSS (spinners, pulsos, transiciones)
  - Panel de estadísticas (docs, chars, última carga)
  - Indicador "pensando..." animado mientras el LLM genera
  - Texto almacenado en Milvus (sin acceso a disco al consultar)
  - Auto-install + fallbacks para PDF
"""
import os
import sys
import subprocess
import time
from datetime import datetime

# ══════════════════════════════════════
# AUTO-INSTALL DE DEPENDENCIAS
# ══════════════════════════════════════
def ensure_pdfminer():
    try:
        from pdfminer.high_level import extract_text
        return True
    except ImportError:
        print("Instalando pdfminer.six...")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', 'pdfminer.six', '-q']
            )
            return True
        except:
            return False

PDFMINER_OK = ensure_pdfminer()

# ══════════════════════════════════════
# PATHS
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
# CONFIG
# ══════════════════════════════════════
COLLECTION = 'rag_documents'
MILVUS_DIR = 'milvus-data'
MAX_TEXT = 1500
EMB_DIM = 384

stats = {
    "last_upload": None,
    "total_chars": 0,
    "session_uploads": 0,
}

# ══════════════════════════════════════
# MILVUS
# ══════════════════════════════════════
def start_milvus():
    try:
        connections.disconnect('default')
    except:
        pass
    default_server.set_base_dir(MILVUS_DIR)
    if not default_server.running:
        default_server.start()
    connections.connect(
        alias='default', host='localhost',
        port=default_server.listen_port
    )

def ensure_collection():
    if utility.has_collection(COLLECTION):
        return Collection(COLLECTION)
    fields = [
        FieldSchema(name='doc_id', dtype=DataType.VARCHAR,
                     max_length=500, is_primary=True, auto_id=False),
        FieldSchema(name='text_content', dtype=DataType.VARCHAR,
                     max_length=2000),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR,
                     dim=EMB_DIM),
    ]
    schema = CollectionSchema(fields=fields)
    col = Collection(name=COLLECTION, schema=schema)
    col.create_index(
        field_name="embedding",
        index_params={'metric_type': 'IP', 'index_type': 'IVF_FLAT',
                      'params': {'nlist': 2048}}
    )
    return col

def get_indexed_docs():
    try:
        col = Collection(COLLECTION)
        col.load()
        count = col.num_entities
        res = col.query(expr="doc_id != ''",
                        output_fields=['doc_id', 'text_content'], limit=200)
        col.release()
        return res, count
    except:
        return [], 0

# ══════════════════════════════════════
# TEXT EXTRACTION (BULLETPROOF)
# ══════════════════════════════════════
def clean_text(text):
    if not text:
        return ""
    import re
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()[:MAX_TEXT]

def extract_pdf(path):
    if PDFMINER_OK:
        try:
            from pdfminer.high_level import extract_text
            t = extract_text(path)
            if t and t.strip():
                return t
        except:
            pass
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        r = PdfReader(path)
        t = '\n'.join(p.extract_text() or '' for p in r.pages)
        if t.strip():
            return t
    except:
        pass
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        return ''.join(chr(b) for b in raw if 32 <= b < 127 or b in (10, 13, 9))
    except:
        return "[No se pudo extraer texto]"

def extract_text_from_file(path):
    if path.lower().endswith('.pdf'):
        return clean_text(extract_pdf(path))
    try:
        with open(path, 'r', errors='ignore') as f:
            return clean_text(f.read())
    except:
        with open(path, 'rb') as f:
            return clean_text(f.read().decode('utf-8', errors='ignore'))

# ══════════════════════════════════════
# INGESTA CON PROGRESO
# ══════════════════════════════════════
def process_uploads(files, progress=None):
    import gradio as gr

    if not files:
        return render_log([("⚠️", "No se seleccionaron archivos", "warn", "")])

    ensure_collection()
    entries = []
    total = len(files)

    if progress is not None:
        progress(0, desc="Preparando archivos...")

    for i, file in enumerate(files):
        path = file.name if hasattr(file, 'name') else str(file)
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()

        if progress is not None:
            progress(i / total, desc=f"Procesando {i+1}/{total}: {name}")

        if ext not in ('.pdf', '.txt', '.text', '.md'):
            entries.append(("⏭️", name, "skip", "Formato no soportado"))
            continue

        try:
            text = extract_text_from_file(path)
            if not text.strip() or len(text.strip()) < 10:
                entries.append(("⚠️", name, "warn", "Sin texto extraíble"))
                continue

            embedding = model_embedding.get_embeddings(text)

            col = Collection(COLLECTION)
            col.insert([[name], [text], [embedding]])
            col.flush()

            chars = len(text)
            stats["total_chars"] += chars
            stats["session_uploads"] += 1
            stats["last_upload"] = datetime.now().strftime("%H:%M:%S")

            entries.append(("✅", name, "ok", f"{chars} caracteres"))

        except Exception as e:
            entries.append(("❌", name, "error", str(e)[:80]))

    if progress is not None:
        progress(1.0, desc="¡Completado!")

    return render_log(entries)


def render_log(entries):
    html_parts = []
    ok = sum(1 for e in entries if e[2] == "ok")
    total = len(entries)

    for idx, (icon, name, status, detail) in enumerate(entries):
        if status == "ok":
            bg = "#E8F5E9"; border = "#4CAF50"; color = "#2E7D32"
        elif status == "error":
            bg = "#FFEBEE"; border = "#E53935"; color = "#C62828"
        elif status == "warn":
            bg = "#FFF8E1"; border = "#FFB74D"; color = "#E65100"
        else:
            bg = "#F5F5F5"; border = "#BDBDBD"; color = "#616161"

        html_parts.append(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
                    margin:6px 0;border-radius:10px;background:{bg};
                    border-left:4px solid {border};
                    animation:slideIn 0.3s ease-out both;
                    animation-delay:{idx*0.1}s">
            <span style="font-size:18px">{icon}</span>
            <div style="flex:1">
                <div style="font-weight:600;color:{color};font-size:14px">{name}</div>
                <div style="font-size:12px;color:#666">{detail}</div>
            </div>
        </div>
        """)

    if ok == total and total > 0:
        summary_bg = "#E8F5E9"; summary_icon = "🎉"; summary_text = "¡Todos los archivos indexados!"
    elif ok > 0:
        summary_bg = "#FFF8E1"; summary_icon = "⚡"; summary_text = f"{ok}/{total} archivos indexados"
    elif total > 0:
        summary_bg = "#FFEBEE"; summary_icon = "😞"; summary_text = "No se pudo indexar ningún archivo"
    else:
        summary_bg = "#F5F5F5"; summary_icon = "📭"; summary_text = "Sin archivos"

    summary_html = f"""
    <div style="margin-top:12px;padding:14px 18px;border-radius:12px;
                background:{summary_bg};text-align:center;
                animation:slideIn 0.4s ease-out both;
                animation-delay:{len(entries)*0.1}s">
        <span style="font-size:24px">{summary_icon}</span>
        <div style="font-weight:700;font-size:16px;margin-top:4px">{summary_text}</div>
    </div>
    """

    return f"""
    <style>
        @keyframes slideIn {{
            from {{ opacity:0; transform:translateX(-12px); }}
            to {{ opacity:1; transform:translateX(0); }}
        }}
    </style>
    {"".join(html_parts)}
    {summary_html}
    """

# ══════════════════════════════════════
# STATS PANEL
# ══════════════════════════════════════
def get_stats_html():
    docs, count = get_indexed_docs()
    total_chars = sum(len(d.get('text_content', '')) for d in docs)
    last = stats.get("last_upload", "—")

    doc_items = ""
    for d in sorted(docs, key=lambda x: x['doc_id']):
        name = d['doc_id']
        chars = len(d.get('text_content', ''))
        icon = "📄" if name.lower().endswith('.pdf') else "📝"
        doc_items += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
                    border-radius:8px;background:#F7F9FB;margin:4px 0;
                    border:1px solid #E2E8F0;font-size:13px;
                    transition:all 0.2s;cursor:default"
             onmouseover="this.style.background='#EAF3F7';this.style.borderColor='#B0D8E8'"
             onmouseout="this.style.background='#F7F9FB';this.style.borderColor='#E2E8F0'">
            <span>{icon}</span>
            <span style="flex:1;font-weight:500;color:#1A2332">{name}</span>
            <span style="color:#5A6A7A;font-size:11px">{chars} chars</span>
        </div>
        """

    if not doc_items:
        doc_items = """
        <div style="text-align:center;padding:32px;color:#9E9E9E">
            <div style="font-size:36px;margin-bottom:8px">📂</div>
            <div>No hay documentos</div>
            <div style="font-size:12px;margin-top:4px">Sube archivos en 📤</div>
        </div>
        """

    return f"""
    <style>
        @keyframes countUp {{
            from {{ opacity:0; transform:translateY(8px); }}
            to {{ opacity:1; transform:translateY(0); }}
        }}
    </style>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px">
        <div style="background:white;border-radius:12px;padding:14px 16px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both">
            <div style="font-size:28px;font-weight:700;color:#F96702;line-height:1.2">{count}</div>
            <div style="font-size:11px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Documentos</div>
        </div>
        <div style="background:white;border-radius:12px;padding:14px 16px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both;animation-delay:0.1s">
            <div style="font-size:28px;font-weight:700;color:#1B5E7B;line-height:1.2">{total_chars:,}</div>
            <div style="font-size:11px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Caracteres</div>
        </div>
        <div style="background:white;border-radius:12px;padding:14px 16px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both;animation-delay:0.2s">
            <div style="font-size:16px;font-weight:700;color:#1B5E7B;padding-top:6px">{last}</div>
            <div style="font-size:11px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Última carga</div>
        </div>
    </div>
    <div style="max-height:340px;overflow-y:auto;padding-right:4px">
        {doc_items}
    </div>
    """

# ══════════════════════════════════════
# RAG QUERY
# ══════════════════════════════════════
def chat_query(question, history):
    if not question.strip():
        return "", history, get_stats_html()

    thinking_msg = "🔍 Buscando documentos relevantes..."
    history = history + [(question, thinking_msg)]
    yield "", history, get_stats_html()

    try:
        col = Collection(COLLECTION)
        col.load()

        q_emb = model_embedding.get_embeddings(question)
        results = col.search(
            data=[q_emb],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=1,
            output_fields=['doc_id', 'text_content'],
            consistency_level="Strong"
        )
        col.release()

        if not results or not results[0]:
            history[-1] = (question, "No encontré documentos relevantes. Sube archivos en la pestaña 📤")
            yield "", history, get_stats_html()
            return

        hit = results[0][0]
        doc_name = hit.entity.get('doc_id')
        context = hit.entity.get('text_content')
        score = hit.distance

        history[-1] = (question, f"📄 Encontrado: *{doc_name}*\n\n🤖 Generando respuesta...")
        yield "", history, get_stats_html()

        prompt = f"""<human>:{context}. Answer this question based on given context {question}
<bot>:"""
        response = model_llm.get_llm_generation(
            prompt, ['<human>:', '\n<bot>:'],
            max_new_tokens=256, do_sample=False,
            temperature=0.7, top_p=0.85,
            top_k=70, repetition_penalty=1.07
        )

        source_badge = f"\n\n---\n📎 **Fuente:** {doc_name} · Relevancia: {score:.0%}"
        history[-1] = (question, response + source_badge)
        yield "", history, get_stats_html()

    except Exception as e:
        history[-1] = (question, f"❌ Error: {str(e)}")
        yield "", history, get_stats_html()

# ══════════════════════════════════════
# CSS
# ══════════════════════════════════════
APP_CSS = """
.header-box {
    background: linear-gradient(135deg, #0D3D56 0%, #1B5E7B 50%, #2A8CB5 100%);
    padding: 28px 32px; border-radius: 16px; margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(13,61,86,0.18);
    position: relative; overflow: hidden;
}
.header-box::before {
    content: ''; position: absolute; top: -40%; right: -15%;
    width: 300px; height: 300px; border-radius: 50%;
    background: rgba(249,103,2,0.08);
}
.header-box h1 {
    color: white !important; font-size: 28px !important;
    margin: 0 0 6px !important; position: relative; z-index: 1;
}
.header-box p {
    color: rgba(255,255,255,0.75) !important; font-size: 14px !important;
    margin: 0 !important; position: relative; z-index: 1;
}
.accent { color: #F96702 !important; }
.tab-nav { border-bottom: 2px solid #E2E8F0 !important; }
.tab-nav button {
    font-weight: 600 !important; font-size: 15px !important;
    padding: 14px 22px !important; transition: all 0.2s !important;
}
.tab-nav button.selected {
    border-bottom: 3px solid #F96702 !important; color: #1B5E7B !important;
}
.upload-zone {
    border: 2px dashed #B0D8E8 !important; border-radius: 14px !important;
    background: #F7FBFD !important; transition: all 0.3s !important;
}
.upload-zone:hover {
    border-color: #F96702 !important; background: #FFF8F3 !important;
}
.footer { text-align:center; padding:20px; color:#9E9E9E; font-size:12px; }
"""

# ══════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════
def create_app():
    import gradio as gr

    with gr.Blocks(title="RAG Chatbot | Cloudera AI") as app:

        gr.HTML("""
        <div class="header-box">
            <h1>💬 RAG Chatbot <span class="accent">| Cloudera AI</span></h1>
            <p>Sube documentos · Haz preguntas · Obtén respuestas inteligentes</p>
        </div>
        """)

        with gr.Tabs():
            # ═══ TAB: CHAT ═══
            with gr.Tab("💬 Chatbot"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=500, show_label=False,
                        )
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Escribe tu pregunta aquí...",
                                show_label=False, scale=5, container=False,
                            )
                            send = gr.Button("Enviar ➤", variant="primary", scale=1)
                        clear = gr.Button("🗑️ Limpiar chat", size="sm")

                    with gr.Column(scale=1, min_width=280):
                        stats_panel = gr.HTML(
                            value="<div style='padding:20px;text-align:center;color:#999'>Cargando...</div>",
                        )
                        refresh = gr.Button("🔄 Actualizar", size="sm")

                gr.Markdown("### 💡 Ejemplos")
                gr.Examples(
                    examples=[
                        ["What are ML Runtimes?"],
                        ["How do data scientists use CML?"],
                        ["What are iceberg tables?"],
                    ],
                    inputs=msg,
                )

            # ═══ TAB: UPLOAD ═══
            with gr.Tab("📤 Subir Documentos"):
                gr.Markdown("""
                ### Agregar Documentos a la Base de Conocimiento
                Arrastra archivos **PDF** o **TXT** y haz clic en procesar.
                El sistema extrae el texto, lo limpia, genera embeddings, y lo almacena en Milvus.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(
                            label="Arrastra archivos aquí",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".text", ".md"],
                            elem_classes=["upload-zone"],
                            height=220,
                        )
                        upload_btn = gr.Button(
                            "🚀 Procesar e Indexar",
                            variant="primary", size="lg",
                        )
                    with gr.Column(scale=1):
                        upload_log = gr.HTML(
                            value="""
                            <div style="text-align:center;padding:60px 20px;color:#BDBDBD">
                                <div style="font-size:48px;margin-bottom:12px">📤</div>
                                <div style="font-size:15px">Los resultados aparecerán aquí</div>
                            </div>
                            """,
                        )

                gr.Markdown("""
                ---
                **Formatos:** PDF, TXT, MD · **Máx recomendado:** 10 MB por archivo ·
                **Auto-instalación:** pdfminer se instala automáticamente si no está disponible
                """)

            # ═══ TAB: ABOUT ═══
            with gr.Tab("ℹ️ Arquitectura"):
                gr.Markdown("""
                ### ¿Cómo funciona?

                **Al subir un documento:**

                ```
                📄 Documento (PDF/TXT)
                  ↓ pdfminer → pypdf → raw (3 fallbacks)
                📝 Texto extraído
                  ↓ Limpiar UTF-8 + truncar a 1500 chars
                🔢 Embedding (all-MiniLM-L6-v2, 384-dim)
                  ↓
                🗄️ Milvus: guarda texto + vector juntos
                ```

                **Al hacer una pregunta:**

                ```
                ❓ Pregunta
                  ↓ Embedding
                🔍 Búsqueda en Milvus
                  ↓ Texto directo (sin leer disco)
                🤖 LLM + contexto → Respuesta
                ```

                ---

                ### ¿Por qué es robusto?

                | Problema | Solución |
                |---|---|
                | pdfminer no instalado | Auto-instala + 2 fallbacks |
                | Bytes UTF-8 inválidos | Limpieza al ingestar |
                | CUDA out of memory | Truncado a 1500 chars |
                | Archivo borrado después de indexar | Texto vive en Milvus |
                | Interfaz se congela | Progreso visual por archivo |
                """)

        gr.HTML('<div class="footer">Cloudera Solutions Engineering · Powered by Open-Source AI · 2026</div>')

        # ═══ EVENTS ═══
        send.click(chat_query, [msg, chatbot], [msg, chatbot, stats_panel])
        msg.submit(chat_query, [msg, chatbot], [msg, chatbot, stats_panel])
        clear.click(lambda: ([], "", get_stats_html()), outputs=[chatbot, msg, stats_panel])
        upload_btn.click(process_uploads, [files], [upload_log]).then(
            lambda: get_stats_html(), outputs=[stats_panel]
        )
        refresh.click(lambda: get_stats_html(), outputs=[stats_panel])
        app.load(lambda: get_stats_html(), outputs=[stats_panel])

    return app

# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════
if __name__ == "__main__":
    import gradio as gr
    print("RAG Chatbot v4 iniciando...")
    start_milvus()
    ensure_collection()
    print("Milvus listo")
    app = create_app()
    app.queue().launch(
        share=False, show_error=True,
        server_name='127.0.0.1',
        server_port=int(os.getenv('CDSW_APP_PORT', '8080')),
        css=APP_CSS,
    )
