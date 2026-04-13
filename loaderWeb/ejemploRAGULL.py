from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage._lc_store import create_kv_docstore

CHROMA_DIR = "./chromadb_ull"
COLLECTION_NAME = "ull"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434"
TOP_K = 4


@dataclass
class Contexto:
    retriever: VectorStoreRetriever


# =========================
# 1) EMBEDDINGS
# =========================
def crear_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )


# =========================
# 2) CREAR RETRIEVER
# =========================
def crear_retriever_simple(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

def crear_retriever_padre_hijo(vectorstore:Chroma):

    ls = LocalFileStore("./padres_store")
    docstore = create_kv_docstore(ls)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": TOP_K},  
    )

    return retriever


@tool
def buscarInformacionPP(query: str, contexto: ToolRuntime[Contexto]):
    """
    Ua esta herramienta cuando el usuario quiera obtener información académica.
    """
    documentos = contexto.context.retriever.invoke(query)
    return documentos


def main():
    embeddings = crear_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    retriever = crear_retriever_padre_hijo(vectorstore)

    print("RAG web listo")

    print("Escribe 'end' para terminar.\n")

    modelo = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        reasoning=True,
    )

    agente = create_agent(
        model=modelo,
        system_prompt="""
        Eres un asistente respetuoso y atento que llama a herramientas.
        Usa la herramienta cuando el usuario tenga dudas sobre cualquier tipo de información académica sobre el grado
        de ingeniería informática
        Responde en espanol.
        Si no encuentras la respuesta en el contexto, dilo claramente.
        Siempre cita al final la URL o las URLs exactas de donde sale la respuesta.
        """,
        tools=[buscarInformacionPP],
        context_schema=Contexto,
    )

    while (prompt := input("> ")) != "end":
        for paso in agente.stream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="values",
            context=Contexto(retriever=retriever),
        ):
            ultimo_mensaje = paso["messages"][-1]

            razonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"):
                razonamiento = ultimo_mensaje.additional_kwargs.get(
                    "reasoning_content", ""
                )

            if razonamiento:
                print("\n=== PENSANDO ===")
                print(razonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()


if __name__ == "__main__":
    main()
