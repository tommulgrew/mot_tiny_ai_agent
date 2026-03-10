Memory service design
================================================================================

- "Background" job with own task queue.
- Agent submits conversation fragments to extract memories from.
    - LLM call used.
    - Memory = facts + keywords
    - Keywords are stemmed and converted to lowercase
- Agent queries memory service for memories matching conversation fragment
    - No LLM used!
    - Words extracted, stemmed and converted to lowercase
    - Matching memories found
    - First random 8 memories returned

### Need to also capture

- Unique ID. Can use incrementing integer
- Date and time memory was created (so that obsolete memories can be identified)
- Hit and miss counts for keywords

### Housekeeping

- Triggered by query
- Submit housekeeping task with conversation fragment, keywords, and extracted memories (full objects)
- Pass to LLM call with instructions to classify each memory as on of the following categories:
    - Duplicate of XXX
    - Contradicts XXX
    - Expired
    - Relevant (to conversation snippet)
    - Not relevant (to conversation snippet)
- Pass IDs in for LLM to refer to.

- Memory service uses LLM classification to:
    - Merge duplicates
    - Delete memories that are contradicted by newer ones
    - Delete expired memories
    - Count keyword hits and misses

