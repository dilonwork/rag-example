# Chunking Strategies

Chunking decides how documents are split before embedding, and it has an
outsized effect on retrieval quality. The simplest approach is **fixed-size
chunking**: cut the text every N characters with some overlap so that ideas
spanning a boundary are not lost. Overlap of 10–20% of the chunk size is a
common starting point.

Better approaches respect document structure. **Semantic chunking** splits on
meaning — for example at sentence or paragraph boundaries, or where the topic
shifts. **Recursive chunking** tries large separators first (double newlines,
then single newlines, then spaces) and falls back to smaller ones.

Two practical rules: keep chunks small enough that one chunk holds one idea
(200–800 characters is typical for Q&A), and always store metadata — document
title, source path, chunk index — so answers can cite where they came from.
If retrieval quality is poor, chunking is the first thing to experiment with,
before touching the embedding model.
