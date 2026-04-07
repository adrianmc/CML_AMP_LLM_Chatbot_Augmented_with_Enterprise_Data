from milvus import default_server
from pymilvus import connections, utility
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

        # Cargar TXTs
        for file in Path(doc_dir).glob('**/*.txt'):
            with open(file, "r") as f:
                print(f"Embedding: {file.name}")
                insert_embedding(collection, os.path.abspath(file), f.read())
                count += 1

        # Cargar PDFs
        for file in Path(doc_dir).glob('**/*.pdf'):
            print(f"Embedding PDF: {file.name}")
            text = extract_text(str(file))
            insert_embedding(collection, os.path.abspath(file), text)
            count += 1

        collection.flush()
        print(f'Total embeddings: {collection.num_entities}')

    except Exception as e:
        default_server.stop()
        raise e

    default_server.stop()

if __name__ == "__main__":
    main()