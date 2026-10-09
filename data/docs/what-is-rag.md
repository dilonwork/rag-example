# What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that grounds a large language
model's answers in external documents. Instead of relying only on what the model
memorized during training, a RAG system retrieves relevant passages from a
knowledge base at query time and feeds them to the model as context.

A typical RAG pipeline has two phases. In the **indexing phase**, documents are
loaded, split into chunks, converted into dense vector embeddings, and stored in
a vector database. In the **query phase**, the user's question is embedded with
the same model, the most similar chunks are retrieved, and a prompt combining
the question with the retrieved context is sent to the LLM.

The main benefits of RAG are factual accuracy, citations to real sources, and
the ability to answer questions about private or up-to-date documents that the
model never saw during training. Its main failure modes are bad chunking,
weak retrieval, and prompts that let the model ignore the context.
