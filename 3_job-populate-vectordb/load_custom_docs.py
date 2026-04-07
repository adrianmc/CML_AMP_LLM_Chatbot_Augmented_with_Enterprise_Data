from milvus import default_server
from pymilvus import connections, utility
import sys
sys.path.insert(0, '/home/cdsw/3_job-populate-vectordb')
from vectordb_insert import create_milvus_collection, insert_embedding
from pdfminer.high_level import extract_text
from pathlib import Path
import subprocess, os

def main():
    subprocess.run(["rm", "-rf", "milvus-data"])
    default_server.set_base_dir('milvus-data')
    default_server.start()

    try:
        connections.connect(alias='default', host='localhost', port=default_server.listen_port)
        collection = create_milvus_collection('cloudera_ml_docs', 384)
        print("Collection created")

        doc_dir = './data'
        count = 0

        # Cargar archivos TXT originales
        for txt_file in Path(doc_dir).glob('**/*.txt'):
            with open(txt_file, 'r', errors='ignore') as fh:
                print(f'Embedding TXT: {txt_file.name}')
                text = fh.read()
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
                insert_embedding(
                    collection, str(txt_file.absolute()), text
                )
                count += 1

        # Cargar archivos PDF custom
        for pdf_file in Path(doc_dir).glob('**/*.pdf'):
            print(f'Embedding PDF: {pdf_file.name}')
            text = extract_text(str(pdf_file))
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            insert_embedding(
                collection, str(pdf_file.absolute()), text
            )
            count += 1
        collection.flush()
        print(f'Total embeddings: {collection.num_entities}')

    except Exception as e:
        default_server.stop()
        raise e

    default_server.stop()

if __name__ == "__main__":
    main()