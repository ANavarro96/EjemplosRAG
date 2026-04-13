"""
Usaremos este codigo para crear la base de datos vectorial a partir de varias webs.
"""

from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import WebBaseLoader, PlaywrightURLLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage._lc_store import create_kv_docstore


URLS = [
    "https://www.ull.es/grados/ingenieria-informatica/",
    "https://www.ull.es/grados/ingenieria-informatica/informacion-academica/horarios-y-calendario-examenes/",
    "https://www.ull.es/grados/ingenieria-informatica/plan-de-estudios/estructura-del-plan-de-estudios/",
]
CHROMA_DIR = "./chromadb_ull"
COLLECTION_NAME = "ull"

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

DOC_PADRE = LocalFileStore("./padres_store")

def cargar_documentos(urls: list[str]):
    loader = WebBaseLoader(
    web_paths=urls,
)
    documentos = loader.load()

    return documentos


# Vamos a probar con la técnica del padre y del hijo
def creamos_splitters():

    # Initialize text splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, add_start_index=True)

    return parent_splitter, child_splitter


def crear_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )


def crear_vectorstore(embeddings, padre, hijo, documentos):
    """
    Si la coleccion ya existe en disco, la reutiliza.
    Si no existe, indexa los documentos.
    """
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore = create_kv_docstore(DOC_PADRE) ,
        child_splitter=hijo,
        parent_splitter=padre,
    )

    num_docs = vectorstore._collection.count()

    if num_docs == 0:
    

        print("Guardamos documentos web en Chroma...")
        retriever.add_documents(documentos)
        print(f"Chunks indexados: {len(documentos)}")
    else:
        print(f"Ya tenemos este numero de documentos indexados: {num_docs}")

    return vectorstore


def main():
    documentos = cargar_documentos(URLS)

    print("Documentos web cargados...")
    padre,hijo = creamos_splitters()
    print("...Documentos partidos...")

    embeddings = crear_embeddings()
    print("...LLM para embeddings listo...")

    crear_vectorstore(embeddings, padre, hijo, documentos)
    print("Ya tenemos chromadb creado con los datos de las web de los partidos políticos")


if __name__ == "__main__":
    main()
