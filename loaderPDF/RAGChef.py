from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
from dataclasses import dataclass
import unicodedata


@dataclass
class Contexto:
    retriever: VectorStoreRetriever


PDF_PATH = "ficheros/vegano_sin_indice.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "recetario_vegano"

#EMBED_MODEL = "mxbai-embed-large"
EMBED_MODEL = "nomic-embed-text"

# Modelo generativo en Ollama para responder
LLM_MODEL = "gemma4:e2b"   # cambia por el que tengas instalado


def cargar_documentos(fichero: str):
    loader = PyPDFLoader(fichero)
    documentos = loader.load()
    return documentos


def _obtener_lineas(texto: str) -> list[str]:
    return [linea.strip() for linea in texto.splitlines() if linea.strip()]


def _normalizar_texto(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")


def _es_pagina_continuacion(lineas: list[str]) -> bool:
    if not lineas:
        return False
    return _normalizar_texto(lineas[0]).startswith("preparacion")


def _obtener_titulo_receta(lineas: list[str]) -> str:
    if not lineas:
        return "Receta sin título"

    titulo = lineas[0]

    # Algunas recetas tienen el título partido en dos líneas, por ejemplo:
    # "Tomates rellenos con tofunesa de" + "remolacha".
    if len(lineas) > 1:
        segunda_linea = lineas[1]
        empieza_seccion = segunda_linea.startswith("Ingredientes") or segunda_linea.startswith("Preparación")
        es_continuacion_titulo = segunda_linea[:1].islower()

        if not empieza_seccion and es_continuacion_titulo:
            titulo = f"{titulo} {segunda_linea}"

    return titulo


def partir_documentos_propio(documentos):
    chunks = []
    receta_actual = []
    metadata_actual = None

    for documento in documentos:
        contenido = documento.page_content.strip()
        # lineas = _obtener_lineas(contenido)
        lineas = contenido.splitlines()

        # Las portadas de sección del PDF no aportan texto útil para recuperar.
        if not lineas:
            continue

        # Miramos a ver si la receta es una continuación de la página anterior o no.
        es_nueva_receta = not (lineas[0]).lower().startswith("preparación")

        if es_nueva_receta:
        # Me la guaardo y seguimos con la siguiente página
            if receta_actual and metadata_actual is not None:
                chunks.append(
                    Document(
                        page_content="\n\n".join(receta_actual).strip(),
                        metadata=metadata_actual,
                    )
                )

            # Todo esto son metadatos, la verdad es que no es obligatorio #
            titulo = _obtener_titulo_receta(lineas)
            metadata_actual = dict(documento.metadata)
            metadata_actual["chunk_type"] = "recipe"
            metadata_actual["recipe_title"] = titulo
            metadata_actual["page_start"] = documento.metadata.get("page")
            metadata_actual["page_end"] = documento.metadata.get("page")
            receta_actual = [contenido]
            continue

        # si no la es, me la guardo en una variable axiliar y sigo palante
        receta_actual.append(contenido)
        metadata_actual["page_end"] = documento.metadata.get("page")

    # Me guardo la última
    if receta_actual and metadata_actual is not None:
        chunks.append(
            Document(
                page_content="\n\n".join(receta_actual).strip(),
                metadata=metadata_actual,
            )
        )

    return chunks


def partir_documentos(documentos):
    return partir_documentos_propio(documentos)


def crear_embeddings():

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url="http://localhost:11434",
    )
    return embeddings


def crear_vectorstore(embeddings, chunks=None):
    """
    Si la colección ya existe en disco, la reutiliza.
    Si no existe, indexa los documentos.
    """

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # vectorstore.reset_collection()
    num_docs = vectorstore._collection.count()

    if num_docs == 0:
        print("Guardamos documentos en Chroma")
        vectorstore.add_documents(chunks)
    else:
        print(f"Ya tenemos este número de documentos: {num_docs}")

    return vectorstore


def crear_retriever(vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever


@tool
def obtenerReceta(query: str, contexto: ToolRuntime[Contexto]):
    """
    Llama a esta herramienta siempre que el usuario quiera obtener información sobre algún tipo de receta vegana.
    Usala cuando el usuario nombre alguna verdura o plato típico vegetariano o vegano.
    """

    vectorstore = contexto.context.retriever

    return vectorstore.invoke(query)


# Añadir al power:
# Si falla por falta de contexto, sube a 700–800. Si recupera demasiado ruido, baja a 300–400.
def main():
    documentos = cargar_documentos(PDF_PATH)
    chunks = partir_documentos_propio(documentos)
    print(f"Recetas detectadas: {len(chunks)}")

    embeddings = crear_embeddings()
    vectorstore = crear_vectorstore(embeddings, chunks)
    retriever = crear_retriever(vectorstore)

    print("RAG listo. Escribe una pregunta sobre el PDF.")
    print("Escribe 'salir' para terminar.\n")

    modelo = ChatOllama(model=LLM_MODEL, reasoning=True)

    agente = create_agent(
        model=modelo,
        system_prompt='''
    Eres un asistente respetuoso y atento, que llama a herramientas.
    Vas a llamar a las herramientas cuando el usuario te pregunte sobre recetas veganas.
    No inventes la salida.
    No cambies de tema.
    Si la herramienta devuelve una o más recetas, ponlas en español.
    ''',
        tools=[obtenerReceta],
        context_schema=Contexto,
    )

    while (prompt := input("> ")) != "end":
        for paso in agente.stream(
            {"messages": [HumanMessage(prompt)]},
            stream_mode="values",
            context=Contexto(retriever=retriever),
        ):
            ultimo_mensaje = paso["messages"][-1]

            hay_razonamiento = ""
            if hasattr(ultimo_mensaje, "additional_kwargs"):
                hay_razonamiento = ultimo_mensaje.additional_kwargs.get("reasoning_content", "")

            if hay_razonamiento:
                print("\n=== PENSANDO ===")
                print(hay_razonamiento)

            print("\n=== MENSAJE ===")
            ultimo_mensaje.pretty_print()


if __name__ == "__main__":
    main()
