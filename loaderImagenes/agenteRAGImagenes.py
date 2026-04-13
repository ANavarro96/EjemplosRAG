import os
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
import base64
import io
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

CARPETA_IMAGENES = os.path.join(PROJECT_ROOT, "imagenes")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db_imagenes")
COLLECTION_NAME = "pokemones"
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".webp")


def cargar_imagenes_en_vectorstore(carpeta_imagenes: str):
    if not os.path.isdir(carpeta_imagenes):
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {carpeta_imagenes}")

    embeddings =OpenCLIPEmbeddings(
        model_name="ViT-B-32",
        checkpoint="openai"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    uris = []
    metadatas = []

    for nombre in sorted(os.listdir(carpeta_imagenes)):
        if nombre.lower().endswith(EXTENSIONES_VALIDAS):
            ruta = os.path.abspath(os.path.join(carpeta_imagenes, nombre))
            uris.append(ruta)
            metadatas.append({
                "filename": nombre,
                "ruta": ruta
            })

    if not uris:
        raise ValueError("No se encontraron imágenes en la carpeta.")

    # Y poco más!
    if vectorstore._collection.count() == 0:
        vectorstore.add_images(uris=uris, metadatas=metadatas)
        print(f"Se han cargado {len(uris)} imágenes en el vector store.")
    else:
        print("Ya existían documentos en la base de datos.")
    return vectorstore

def mostrar_resultados(resultados):
    for doc in resultados:
        print(f"Documento obtenido: {doc}")
        ruta = doc.metadata["ruta"] # esto es porque lo he puesto en los metadatos


        #img = plt.imshow(ruta)
        img_bytes = base64.b64decode(doc.page_content)
        img = Image.open(io.BytesIO(img_bytes))
        plt.imshow(img)
        plt.title(doc.metadata["filename"])
        plt.axis("off")
        plt.show()

def main():
    #vectorstore = cargar_imagenes_en_vectorstore(CARPETA_IMAGENES)

    
    embeddings =OpenCLIPEmbeddings(
    model_name="ViT-B-32",
    checkpoint="openai"
)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    consulta = "pokémon de color amarillo"
    resultados = vectorstore.similarity_search(consulta, k=2)

    mostrar_resultados(resultados)


if __name__ == "__main__":
    main()