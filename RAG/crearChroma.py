"""
Usaremos este código para crear la base de datos vectorial.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "streamers"

"""
Con esta función cargamos los documentos, pasamos de fichero a Documents
"""
def cargar_documentos(fichero: str):
    documentos = open(fichero, "r").readlines() # cargamos el documento
    return documentos


"""
Partimos los documentos en trozos
"""
def partir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(documentos) # como el origen es textual, creo los documentos.
    return chunks


"""
Creamos los embeddings
"""
def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", # El modelo LLM a usar
        base_url="http://localhost:11434", # Esta es la URL de Ollama (local)
    )
    return embeddings


"""
Añadimos los embeddings a Chroma

"""
def crear_vectorstore(embeddings,chunks = None):
    """
    Si la colección ya existe en disco, la reutiliza.
    Si no existe, indexa los documentos.
    """

    # Podéis usar este nétodo también, pero con menos control.
    # Si ya existe la colección la va a duplicar, así que
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    num_docs = vectorstore._collection.count()

    if num_docs == 0:
        print("Guardamos documentos en Chroma")
        vectorstore.add_documents(chunks)
    else:
        print(f"Ya tenemos este número de documentos: {num_docs}")

    return vectorstore

def main():
    documentos = cargar_documentos("./RAG/xokas.txt")

    print("Documentos creados...")
    chunks = partir_documentos(documentos)
    print("...Documentos partidos...")

    embeddings = crear_embeddings()

    print("...LLM para embeddings listo...")

    crear_vectorstore(embeddings, chunks)

    print("Ya tenemos chromadb creado!!")

if __name__ == "__main__":
    main()