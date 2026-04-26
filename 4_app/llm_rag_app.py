"""
llm_rag_app.py — RAG Chatbot v5
Cloudera Machine Learning
Compatible con Gradio 6.0+

v5 mejoras sobre v4:
  - Chunking: documentos divididos en fragmentos de 400 chars con overlap
  - Top-K retrieval: recupera 3 chunks relevantes en vez de 1 documento
  - Prompt estructurado: instrucciones claras al LLM
  - Fallback: si respuesta vacía, muestra el contexto encontrado
  - Stats: muestra docs únicos vs chunks totales
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
COLLECTION = 'rag_chunks_v5'
MILVUS_DIR = 'milvus-data'
CHUNK_SIZE = 400       # Caracteres por chunk
CHUNK_OVERLAP = 80     # Solapamiento entre chunks
TOP_K = 3              # Chunks a recuperar por pregunta
MAX_CONTEXT = 1200     # Máx chars de contexto combinado al LLM
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
        FieldSchema(name='chunk_id', dtype=DataType.INT64,
                     is_primary=True, auto_id=True),
        FieldSchema(name='source_name', dtype=DataType.VARCHAR,
                     max_length=500),
        FieldSchema(name='text_content', dtype=DataType.VARCHAR,
                     max_length=600),
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
        res = col.query(expr="chunk_id >= 0",
                        output_fields=['source_name', 'text_content'], limit=500)
        col.release()
        return res, count
    except:
        return [], 0

def get_doc_names():
    """Retorna lista de nombres únicos de documentos para el dropdown."""
    docs, _ = get_indexed_docs()
    names = sorted(set(d.get('source_name', '') for d in docs))
    return names if names else ["(sin documentos)"]

def delete_document(doc_name):
    """Elimina todos los chunks de un documento específico."""
    import gradio as gr

    if not doc_name or doc_name == "(sin documentos)":
        return (
            render_log([("⚠️", "Selecciona un documento", "warn", "")]),
            gr.update(choices=get_doc_names())
        )

    try:
        col = Collection(COLLECTION)
        col.load()

        # Buscar chunks de este documento
        chunks = col.query(
            expr=f'source_name == "{doc_name}"',
            output_fields=['chunk_id'],
            limit=500
        )

        if not chunks:
            col.release()
            return (
                render_log([("⚠️", doc_name, "warn", "No encontrado")]),
                gr.update(choices=get_doc_names())
            )

        # Eliminar por IDs
        chunk_ids = [c['chunk_id'] for c in chunks]
        col.delete(expr=f"chunk_id in {chunk_ids}")
        col.flush()
        col.release()

        n = len(chunk_ids)
        return (
            render_log([("🗑️", doc_name, "ok", f"Eliminado ({n} chunks)")]),
            gr.update(choices=get_doc_names(), value=None)
        )

    except Exception as e:
        return (
            render_log([("❌", doc_name, "error", str(e)[:80])]),
            gr.update(choices=get_doc_names())
        )

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
    return text.strip()

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide texto en chunks con solapamiento para mejor retrieval."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Intentar cortar en un punto natural (punto, salto de línea)
        if end < len(text):
            # Buscar el último punto o salto de línea en el chunk
            last_break = max(chunk.rfind('. '), chunk.rfind('\n'), chunk.rfind('? '), chunk.rfind('! '))
            if last_break > chunk_size * 0.5:  # Solo si está en la segunda mitad
                chunk = chunk[:last_break + 1]
                end = start + last_break + 1
        chunk = chunk.strip()
        if chunk and len(chunk) > 20:
            chunks.append(chunk)
        start = end - overlap
    return chunks

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
            full_text = extract_text_from_file(path)
            if not full_text.strip() or len(full_text.strip()) < 10:
                entries.append(("⚠️", name, "warn", "Sin texto extraíble"))
                continue

            # Dividir en chunks
            chunks = split_into_chunks(full_text)

            # Generar embeddings e insertar cada chunk
            source_names = []
            text_contents = []
            embeddings = []

            for chunk in chunks:
                emb = model_embedding.get_embeddings(chunk)
                source_names.append(name)
                text_contents.append(chunk[:580])  # Límite del campo VARCHAR
                embeddings.append(emb)

            col = Collection(COLLECTION)
            col.insert([source_names, text_contents, embeddings])
            col.flush()

            total_chars = sum(len(c) for c in chunks)
            stats["total_chars"] += total_chars
            stats["session_uploads"] += 1
            stats["last_upload"] = datetime.now().strftime("%H:%M:%S")

            entries.append(("✅", name, "ok", f"{len(chunks)} chunks · {total_chars} chars"))

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

    # Contar documentos únicos vs chunks
    unique_docs = set(d.get('source_name', '') for d in docs)
    n_docs = len(unique_docs)

    # Doc list agrupado
    doc_items = ""
    doc_chunks = {}
    for d in docs:
        src = d.get('source_name', '?')
        doc_chunks[src] = doc_chunks.get(src, 0) + 1

    for name in sorted(doc_chunks.keys()):
        n_chunks = doc_chunks[name]
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
            <span style="color:#5A6A7A;font-size:11px">{n_chunks} chunks</span>
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
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px">
        <div style="background:white;border-radius:12px;padding:12px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both">
            <div style="font-size:24px;font-weight:700;color:#F96702;line-height:1.2">{n_docs}</div>
            <div style="font-size:10px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Docs</div>
        </div>
        <div style="background:white;border-radius:12px;padding:12px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both;animation-delay:0.1s">
            <div style="font-size:24px;font-weight:700;color:#1B5E7B;line-height:1.2">{count}</div>
            <div style="font-size:10px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Chunks</div>
        </div>
        <div style="background:white;border-radius:12px;padding:12px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both;animation-delay:0.15s">
            <div style="font-size:24px;font-weight:700;color:#1B5E7B;line-height:1.2">{total_chars:,}</div>
            <div style="font-size:10px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Chars</div>
        </div>
        <div style="background:white;border-radius:12px;padding:12px;
                    border:1px solid #E2E8F0;text-align:center;
                    animation:countUp 0.4s ease-out both;animation-delay:0.2s">
            <div style="font-size:14px;font-weight:700;color:#1B5E7B;padding-top:4px">{last}</div>
            <div style="font-size:10px;color:#5A6A7A;text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">Última</div>
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

    thinking_msg = "🔍 Buscando fragmentos relevantes..."
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": thinking_msg},
    ]
    yield "", history, get_stats_html()

    try:
        col = Collection(COLLECTION)
        col.load()

        q_emb = model_embedding.get_embeddings(question)
        results = col.search(
            data=[q_emb],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=TOP_K,
            output_fields=['source_name', 'text_content'],
            consistency_level="Strong"
        )
        col.release()

        if not results or not results[0]:
            history[-1] = {"role": "assistant", "content": "No encontré documentos relevantes. Sube archivos en la pestaña 📤"}
            yield "", history, get_stats_html()
            return

        # Combinar top-K chunks como contexto
        context_parts = []
        sources = set()
        best_score = 0
        for hit in results[0]:
            src = hit.entity.get('source_name', '?')
            txt = hit.entity.get('text_content', '')
            score = hit.distance
            if score > best_score:
                best_score = score
            sources.add(src)
            context_parts.append(txt)

        combined_context = "\n\n".join(context_parts)[:MAX_CONTEXT]
        source_list = ", ".join(sources)

        history[-1] = {"role": "assistant", "content": f"📄 Encontrado en: *{source_list}* ({len(context_parts)} fragmentos)\n\n🤖 Generando respuesta..."}
        yield "", history, get_stats_html()

        # Prompt mejorado y estructurado
        prompt = f"""<human>: You are a helpful assistant. Use ONLY the following context to answer the question. If the context does not contain enough information, say so clearly. Answer in the same language as the question.

CONTEXT:
{combined_context}

QUESTION: {question}

Provide a clear, specific answer based only on the context above.
<bot>:"""

        response = model_llm.get_llm_generation(
            prompt, ['<human>:', '\n<bot>:'],
            max_new_tokens=256, do_sample=False,
            temperature=0.7, top_p=0.85,
            top_k=70, repetition_penalty=1.07
        )

        # Validar respuesta
        response = response.strip()
        if not response or len(response) < 5:
            response = f"No pude generar una respuesta clara. El contexto encontrado fue:\n\n> {combined_context[:300]}..."

        source_badge = f"\n\n---\n📎 **Fuentes:** {source_list} · {len(context_parts)} fragmentos · Relevancia: {best_score:.0%}"
        history[-1] = {"role": "assistant", "content": response + source_badge}
        yield "", history, get_stats_html()

    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
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
                        ["¿Cuáles son los puntos clave del documento?"],
                        ["Resume los hallazgos principales"],
                        ["¿Qué recomendaciones se mencionan?"],
                        ["¿Cuáles son los riesgos identificados?"],
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

                gr.Markdown("---")

                # Sección de eliminación
                gr.Markdown("### 🗑️ Eliminar Documentos")
                with gr.Row():
                    with gr.Column(scale=2):
                        doc_dropdown = gr.Dropdown(
                            label="Seleccionar documento a eliminar",
                            choices=get_doc_names(),
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        delete_btn = gr.Button(
                            "🗑️ Eliminar documento",
                            variant="stop", size="lg",
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
                📝 Texto completo extraído + limpieza UTF-8
                  ↓ Dividir en chunks de 400 chars con 80 de overlap
                🔢 Embedding por cada chunk (all-MiniLM-L6-v2, 384-dim)
                  ↓
                🗄️ Milvus: cada chunk = 1 registro (source + texto + vector)
                ```

                **Al hacer una pregunta:**

                ```
                ❓ Pregunta
                  ↓ Embedding
                🔍 Buscar top 3 chunks más similares en Milvus
                  ↓ Combinar texto de los 3 chunks (máx 1200 chars)
                🤖 Prompt estructurado + contexto combinado → LLM → Respuesta
                ```

                ---

                ### Mejoras de accuracy (v5 vs v4)

                | Antes (v4) | Ahora (v5) |
                |---|---|
                | 1 embedding por documento entero | 1 embedding por chunk de 400 chars |
                | Recupera 1 solo documento | Recupera top 3 chunks más relevantes |
                | Contexto = primeros 1500 chars (portada + índice) | Contexto = las 3 secciones más relevantes |
                | Prompt simple sin instrucciones | Prompt estructurado con instrucciones claras |
                | Respuesta vacía = silencio | Respuesta vacía = muestra el contexto encontrado |
                """)

        gr.HTML('<div class="footer">Cloudera Solutions Engineering · Powered by Open-Source AI · 2026</div>')

        # ═══ EVENTS ═══
        send.click(chat_query, [msg, chatbot], [msg, chatbot, stats_panel])
        msg.submit(chat_query, [msg, chatbot], [msg, chatbot, stats_panel])
        clear.click(lambda: ([], "", get_stats_html()), outputs=[chatbot, msg, stats_panel])

        # Upload → refresh stats + dropdown
        upload_btn.click(
            process_uploads, [files], [upload_log]
        ).then(
            lambda: get_stats_html(), outputs=[stats_panel]
        ).then(
            lambda: gr.update(choices=get_doc_names()), outputs=[doc_dropdown]
        )

        # Delete → refresh stats + dropdown + log
        delete_btn.click(
            delete_document, [doc_dropdown], [upload_log, doc_dropdown]
        ).then(
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
    print("RAG Chatbot v5 iniciando...")
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
