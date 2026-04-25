"""
llm_rag_app.py — RAG Chatbot con Interfaz Moderna
Cloudera Machine Learning

Features:
  - Upload de documentos (PDF/TXT) directo desde la interfaz
  - Ingesta automática a Milvus al subir archivos  
  - Chatbot RAG con visualización de fuentes
  - Panel de documentos indexados
  - Branding Cloudera
"""
import os
import sys
import tempfile

# Paths del proyecto
sys.path.insert(0, '/home/cdsw')
sys.path.insert(0, '/home/cdsw/3_job-populate-vectordb')

from milvus import default_server
from pymilvus import connections, Collection, utility
import utils.model_llm_utils as model_llm
import utils.model_embedding_utils as model_embedding

# ══════════════════════════════════════
# MILVUS SETUP
# ══════════════════════════════════════
COLLECTION_NAME = 'cloudera_ml_docs'
MILVUS_DATA_DIR = 'milvus-data'

def start_milvus():
    """Inicia servidor Milvus y conecta."""
    try:
        connections.disconnect('default')
    except:
        pass
    default_server.set_base_dir(MILVUS_DATA_DIR)
    if not default_server.running:
        default_server.start()
    connections.connect(alias='default', host='localhost', port=default_server.listen_port)
    print("Milvus conectado")

def ensure_collection():
    """Crea la colección si no existe."""
    from pymilvus import FieldSchema, CollectionSchema, DataType
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)
    
    fields = [
        FieldSchema(name='relativefilepath', dtype=DataType.VARCHAR,
                     max_length=1000, is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    index_params = {
        'metric_type': 'IP',
        'index_type': 'IVF_FLAT',
        'params': {'nlist': 2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def get_indexed_docs():
    """Retorna lista de documentos indexados en Milvus."""
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        count = collection.num_entities
        # Query para obtener los file paths
        results = collection.query(
            expr="relativefilepath != ''",
            output_fields=['relativefilepath'],
            limit=100
        )
        collection.release()
        docs = [r['relativefilepath'] for r in results]
        return docs, count
    except Exception as e:
        print(f"Error listando docs: {e}")
        return [], 0

# ══════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════
def extract_text_from_file(file_path: str) -> str:
    """Extrae texto de un archivo PDF o TXT."""
    if file_path.lower().endswith('.pdf'):
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Error extrayendo PDF: {e}")
    else:
        with open(file_path, 'r', errors='ignore') as f:
            text = f.read()
    
    # Limpiar UTF-8
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text

def ingest_file(file_path: str, file_name: str) -> str:
    """Ingesta un archivo individual a Milvus."""
    try:
        text = extract_text_from_file(file_path)
        if not text.strip():
            return f"⚠️ {file_name}: Archivo vacío o sin texto extraíble"
        
        # Generar embedding
        embedding = model_embedding.get_embeddings(text)
        
        # Insertar en Milvus
        collection = Collection(COLLECTION_NAME)
        data = [[file_name], [embedding]]
        collection.insert(data)
        collection.flush()
        
        return f"✅ {file_name}: Indexado ({len(text)} caracteres)"
    except Exception as e:
        return f"❌ {file_name}: Error - {str(e)}"

def process_uploads(files) -> str:
    """Procesa múltiples archivos subidos desde la UI."""
    if not files:
        return "⚠️ No se seleccionaron archivos"
    
    ensure_collection()
    results = []
    
    for file in files:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_name = os.path.basename(file_path)
        
        # Validar extensión
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ('.pdf', '.txt', '.text', '.md'):
            results.append(f"⏭️ {file_name}: Formato no soportado (usar PDF o TXT)")
            continue
        
        result = ingest_file(file_path, file_name)
        results.append(result)
    
    # Resumen
    success = sum(1 for r in results if r.startswith("✅"))
    total = len(results)
    summary = f"\n{'─' * 40}\n📊 Resultado: {success}/{total} archivos indexados\n"
    
    return "\n".join(results) + summary

def refresh_doc_list():
    """Actualiza la lista de documentos para mostrar en la UI."""
    docs, count = get_indexed_docs()
    if not docs:
        return f"📂 No hay documentos indexados", f"Total: 0 documentos"
    
    doc_list = []
    for d in docs:
        name = os.path.basename(d)
        icon = "📄" if name.endswith('.pdf') else "📝"
        doc_list.append(f"{icon} {name}")
    
    doc_text = "\n".join(doc_list)
    return doc_text, f"Total: {count} documentos indexados"

# ══════════════════════════════════════
# RAG QUERY
# ══════════════════════════════════════
def get_nearest_chunk(collection, question):
    """Busca el documento más similar a la pregunta."""
    question_embedding = model_embedding.get_embeddings(question)
    
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=['relativefilepath'],
        consistency_level="Strong"
    )
    
    if not results or not results[0]:
        return "No se encontraron documentos relevantes.", "Sin fuente"
    
    doc_id = results[0].ids[0]
    score = results[0].distances[0]
    source_name = os.path.basename(doc_id)
    
    # Cargar texto del documento
    text = load_doc_text(doc_id)
    return text, f"{source_name} (score: {score:.3f})"

def load_doc_text(id_path):
    """Carga texto del documento, soportando PDF y TXT."""
    # Buscar en uploads temporales o en data/
    possible_paths = [
        id_path,
        os.path.join('/home/cdsw/data', id_path),
        os.path.join('/home/cdsw/data/custom_docs', id_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            text = extract_text_from_file(path)
            return text[:2000]  # Truncar para no saturar GPU
    
    # Si el archivo no existe en disco, retornar mensaje
    return f"[Documento: {os.path.basename(id_path)} - archivo no encontrado en disco]"

def chat_query(question, history):
    """Procesa una pregunta del chatbot RAG."""
    if not question.strip():
        return "", history
    
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        # Buscar contexto
        context, source = get_nearest_chunk(collection, question)
        collection.release()
        
        # Generar respuesta CON contexto
        prompt_with_context = f"""<human>:{context}. Answer this question based on given context {question}
<bot>:"""
        
        stop_words = ['<human>:', '\n<bot>:']
        response = model_llm.get_llm_generation(
            prompt_with_context, stop_words,
            max_new_tokens=256, do_sample=False,
            temperature=0.7, top_p=0.85,
            top_k=70, repetition_penalty=1.07
        )
        
        # Agregar fuente
        full_response = f"{response}\n\n📎 *Fuente: {source}*"
        history = history + [(question, full_response)]
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history = history + [(question, error_msg)]
    
    return "", history

# ══════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════
def create_app():
    import gradio as gr
    
    custom_css = """
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #0D3D56 0%, #1B5E7B 60%, #2A8CB5 100%);
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(13,61,86,0.15);
    }
    .header-container h1 {
        color: white !important;
        font-size: 26px !important;
        margin: 0 0 4px !important;
        font-weight: 700 !important;
    }
    .header-container p {
        color: rgba(255,255,255,0.75) !important;
        font-size: 14px !important;
        margin: 0 !important;
    }
    .header-container .accent { color: #F96702 !important; }
    
    /* Tabs */
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 20px !important;
    }
    .tab-nav button.selected {
        border-bottom: 3px solid #F96702 !important;
        color: #1B5E7B !important;
    }
    
    /* Upload zone */
    .upload-zone {
        border: 2px dashed #B0D8E8 !important;
        border-radius: 12px !important;
        background: #F7FBFD !important;
        transition: all 0.2s !important;
    }
    .upload-zone:hover {
        border-color: #F96702 !important;
        background: #FFF8F3 !important;
    }
    
    /* Chat */
    .chatbot .message {
        border-radius: 12px !important;
    }
    
    /* Doc list */
    .doc-panel {
        background: #F7F9FB;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    
    /* Buttons */
    .primary-btn {
        background: #F96702 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .primary-btn:hover {
        background: #FF8534 !important;
    }
    .secondary-btn {
        background: #1B5E7B !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: white !important;
    }
    
    /* Status */
    .status-box {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        padding: 16px;
        color: #5A6A7A;
        font-size: 12px;
    }
    """
    
    with gr.Blocks(
        title="RAG Chatbot | Cloudera AI",
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="blue",
            font=gr.themes.GoogleFont("DM Sans"),
        ),
        css=custom_css,
    ) as app:
        
        # ── Header ──
        gr.HTML("""
        <div class="header-container">
            <h1>💬 RAG Chatbot <span class="accent">| Cloudera AI</span></h1>
            <p>Sube documentos PDF o TXT y hazle preguntas en lenguaje natural</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            # ══════ TAB 1: CHATBOT ══════
            with gr.Tab("💬 Chatbot", id="chat"):
                with gr.Row():
                    # Chat principal
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversación",
                            height=480,
                            show_label=False,
                            avatar_images=(None, "https://www.cloudera.com/content/dam/www/marketing/media-library/cloudera-favicon.png"),
                            bubble_full_width=False,
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="",
                                placeholder="Escribe tu pregunta aquí...",
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            send_btn = gr.Button(
                                "Enviar",
                                variant="primary",
                                scale=1,
                                elem_classes=["primary-btn"],
                            )
                        clear_btn = gr.Button("🗑️ Limpiar chat", size="sm")
                    
                    # Panel lateral
                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Base de Conocimiento")
                        doc_count = gr.Markdown("Total: 0 documentos")
                        doc_list_display = gr.Textbox(
                            label="Documentos indexados",
                            interactive=False,
                            lines=12,
                            show_label=False,
                            elem_classes=["doc-panel"],
                        )
                        refresh_btn = gr.Button(
                            "🔄 Actualizar lista",
                            size="sm",
                            elem_classes=["secondary-btn"],
                        )
                
                # Ejemplos
                gr.Markdown("### 💡 Preguntas de ejemplo")
                gr.Examples(
                    examples=[
                        ["What are ML Runtimes?"],
                        ["What kinds of users use CML?"],
                        ["How do data scientists use CML?"],
                        ["What are iceberg tables?"],
                    ],
                    inputs=msg_input,
                )
            
            # ══════ TAB 2: UPLOAD ══════
            with gr.Tab("📤 Subir Documentos", id="upload"):
                gr.Markdown("""
                ### Subir Documentos a la Base de Conocimiento
                
                Sube archivos **PDF** o **TXT** para que el chatbot pueda responder preguntas sobre su contenido.
                Los documentos se procesan automáticamente: se extrae el texto, se genera un embedding, y se indexan en Milvus.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_upload = gr.File(
                            label="Arrastra archivos aquí o haz clic para seleccionar",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".text", ".md"],
                            elem_classes=["upload-zone"],
                            height=200,
                        )
                        upload_btn = gr.Button(
                            "🚀 Procesar e Indexar Documentos",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"],
                        )
                    
                    with gr.Column(scale=1):
                        upload_status = gr.Textbox(
                            label="Resultado del procesamiento",
                            interactive=False,
                            lines=15,
                            elem_classes=["status-box"],
                        )
                
                gr.Markdown("""
                ---
                **Formatos soportados:** PDF, TXT, MD  
                **Tamaño recomendado:** Máximo 10 MB por archivo  
                **Documentos ideales:** 5-50 páginas para mejores resultados de RAG
                """)
            
            # ══════ TAB 3: ABOUT ══════
            with gr.Tab("ℹ️ Acerca de", id="about"):
                gr.Markdown("""
                ### Arquitectura RAG (Retrieval-Augmented Generation)
                
                ```
                📄 Documento (PDF/TXT)
                    ↓ pdfminer / open()
                📝 Texto extraído
                    ↓ all-MiniLM-L6-v2
                🔢 Embedding (384 dimensiones)
                    ↓
                🗄️ Milvus Vector DB
                ```
                
                ```
                ❓ Pregunta del usuario
                    ↓ all-MiniLM-L6-v2
                🔢 Embedding de la pregunta
                    ↓ Búsqueda de similitud
                📄 Documento más relevante
                    ↓ + pregunta original
                🤖 LLM genera respuesta contextualizada
                ```
                
                ---
                
                ### Componentes
                
                | Componente | Tecnología |
                |---|---|
                | Vector Store | Milvus (integrado) |
                | Embeddings | all-MiniLM-L6-v2 (384-dim) |
                | LLM | h2ogpt-oig-oasst1-512-6.9b |
                | Interfaz | Gradio |
                | Plataforma | Cloudera Machine Learning (CML) |
                
                ---
                
                **Repositorio:** [github.com/adrianmc/CML_AMP_LLM_Chatbot](https://github.com/adrianmc/CML_AMP_LLM_Chatbot_Augmented_with_Enterprise_Data)  
                **Construido por:** Cloudera Solutions Engineering
                """)
        
        # ── Footer ──
        gr.HTML("""
        <div class="footer-text">
            Cloudera Solutions Engineering · Powered by open-source AI · 2026
        </div>
        """)
        
        # ══════ EVENT HANDLERS ══════
        
        # Chat
        send_btn.click(
            fn=chat_query,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            fn=chat_query,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])
        
        # Upload
        upload_btn.click(
            fn=process_uploads,
            inputs=[file_upload],
            outputs=[upload_status],
        )
        
        # Refresh doc list
        refresh_btn.click(
            fn=refresh_doc_list,
            outputs=[doc_list_display, doc_count],
        )
        
        # Auto-refresh on tab change
        app.load(
            fn=refresh_doc_list,
            outputs=[doc_list_display, doc_count],
        )
    
    return app


# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════
if __name__ == "__main__":
    print("Iniciando RAG Chatbot...")
    
    # Iniciar Milvus
    start_milvus()
    ensure_collection()
    print("Milvus listo")
    
    # Crear y lanzar app
    app = create_app()
    app.queue().launch(
        share=False,
        show_error=True,
        server_name='127.0.0.1',
        server_port=int(os.getenv('CDSW_APP_PORT', '8080')),
    )
