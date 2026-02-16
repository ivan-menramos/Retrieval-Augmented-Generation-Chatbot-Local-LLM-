# Sistema RAG Local con Microsoft Phi-2

Sistema de preguntas y respuestas basado en documentos PDF utilizando una arquitectura Retrieval-Augmented Generation (RAG) con un LLM local.

## Descripción del proyecto

Este proyecto implementa un pipeline completo que:

- Ingresa documentos PDF
- Extrae y fragmenta el texto
- Genera embeddings
- Almacena los vectores en FAISS
- Usa el modelo microsoft/phi-2 para generar respuestas basadas en el contexto recuperado

Todo el sistema funciona localmente sin APIs externas.

---

## Arquitectura

Usuario -> FastAPI - > PDF Loader -> Chunking -> Embeddings -> FAISS -> Retriever -> Phi-2 -> Respuesta

---

## Tecnologías Usadas

- Python
- FastAPI
- LangChain
- FAISS
- HuggingFace
- Transformers
- sentence-transformers/all-MiniLM-L6-v2
- microsoft/phi-2


---

##  Aprendizajes

- Orquestación de modelos causales en RAG
- Búsqueda semántica con embeddings
- Construcción de sistemas QA privados
