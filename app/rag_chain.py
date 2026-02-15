from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from app.retriever import construir_retriever
import torch

def build_rag():

    retriever = construir_retriever()

    model_name = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float32,
        device_map = "auto"
    )

    pipe = pipeline(
        "text-generation",
        model = model,
        max_new_tokens = 250,
        tokenizer = tokenizer,
        temperature = 0.2
    )

    llm = HuggingFacePipeline(pipeline = pipe)
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever
    )

    return qa_chain