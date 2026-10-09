# Vector Databases

A vector database stores embeddings — dense numeric vectors that capture the
meaning of text — and supports fast similarity search over them. Given a query
vector, the database returns the stored vectors closest to it, usually measured
by cosine similarity or dot product.

For prototypes and examples, an in-memory NumPy array with brute-force search
is perfectly fine up to tens of thousands of vectors. Beyond that, approximate
nearest neighbor (ANN) indexes such as HNSW trade a little recall for much
faster queries. Popular options include Chroma and Qdrant for self-hosting,
pgvector if you already run Postgres, and Pinecone as a managed service.

What matters more than the choice of database is the **embedding model**:
the query and the documents must be embedded with the same model, and the
model's training domain should match your documents. A general-purpose model
like all-MiniLM-L6-v2 is a solid default; domain-specific models help for
code, biomedical text, or multilingual corpora.
