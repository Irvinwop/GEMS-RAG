# Knowledge

- Native tool ablations must keep provider continuation protocols inside model adapters, aggregate usage across every provider call, and expose only bounded search catalogs until the model explicitly opens evidence.
- Time-sensitive model catalogs need role-aware migrations from current official provider guidance; preserve endpoint and reasoning behavior instead of blindly replacing every model with the flagship tier.
- Manuscript RAG baselines must use explicit component modes and mode-specific readiness checks; graph expansion is only present when neighbor chunks enter the candidate set before ranking.
- Paper-only RAG integrations must fail open about unavailable artifacts: label deterministic corpus adaptations in result debug metadata and never imply they reproduce unreleased learned components.
- The extracted chunk JSONL has 5,821 rows but only 5,707 IDs; all shared-corpus loaders and exporters must canonicalize the 38 colliding IDs before indexing.
- LPKG publishes planner training/inference code but no trained planner checkpoint; integrations must consume normalized official `generated_predictions.jsonl` plans and must not silently relabel an ordinary single-query retriever as LPKG.
- MegaRAG's upstream `mix_two_step` drops `only_need_context` and always synthesizes an answer; fair generator ablations must run its official hybrid-MMKG and naive-page retrieval branches directly and bypass only the upstream second generation stage.
- MegaRAG hashes page chunks from text only; prepared page manifests need a unique page marker or blank/image-only pages overwrite one another during indexing.
