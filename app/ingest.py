import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
def ingesta_documentos():
    documentos = [] #lista acumuladora

    for file in os.listdir("data/documentos"):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(f"data/documentos/{file}")
            documentos.extend(loader.load())
    print("El documento fue cargado")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 350,
        chunk_overlap = 100
    )

    chunks = splitter.split_documents(documentos)

    print(f"Se crearon {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    """
    Un modelo de 6 capas, 384 dimensiones
    MiniLM es buen balance calidad/costo
    """

    vectorstore = FAISS.from_documents(chunks,embeddings)

    vectorstore.save_local("vectorstore")

    print("El vector store fue guardado exitosamente")

if __name__ == "__main__":
    ingesta_documentos()