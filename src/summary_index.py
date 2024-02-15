import time
from llama_index.llms import AzureOpenAI #as llama_openai
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index import StorageContext, VectorStoreIndex, DocumentSummaryIndex, SimpleDirectoryReader, ServiceContext
from llama_index import get_response_synthesizer
import logging
import getpass
import sys
from llama_index import set_global_service_context

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from llama_index.vector_stores import CognitiveSearchVectorStore
from llama_index.vector_stores.cogsearch import (
    IndexManagement,
    MetadataIndexFieldType,
    CognitiveSearchVectorStore,
)
from llama_index import StorageContext, load_index_from_storage


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key=""
azure_endpoint="https://ps-openai-dev.openai.azure.com"
api_version="2023-07-01-preview"
embedding_api_version="2023-05-15"

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="gpt-35-turbo-16k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=embedding_api_version,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
)

set_global_service_context(service_context)

# Add data to the index
#storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(embed_model=embed_model, chunk_size=1024)

documents = SimpleDirectoryReader(input_files=f"test-data/cities-info/Toronto.txt"
    #input_files=[f"test-data/paul_graham_essay.txt"]
).load_data()

documents[0].doc_id = "Toronto"

# default mode of building the index
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    documents,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.get_document_summary("Toronto")

query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("What are the sports teams in Toronto?")
print(response)

from llama_index.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)

retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
    # choice_select_prompt=None,
    # choice_batch_size=10,
    # choice_top_k=1,
    # format_node_batch_fn=None,
    # parse_choice_select_answer_fn=None,
    # service_context=None
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

print(len(retrieved_nodes))
print(retrieved_nodes[0].score)
print(retrieved_nodes[0].node.get_text())

# use retriever as part of a query engine
from llama_index.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(response)


from llama_index.indices.document_summary import (
    DocumentSummaryIndexEmbeddingRetriever,
)

retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
    # similarity_top_k=1,
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")
len(retrieved_nodes)
print(retrieved_nodes[0].node.get_text())

# use retriever as part of a query engine
from llama_index.query_engine import RetrieverQueryEngine

# configure response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# query
response = query_engine.query("What are the sports teams in Toronto?")
print(response)