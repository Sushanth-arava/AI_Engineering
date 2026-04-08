from .rag_pipeline import RAGPipeline

# RAGEvaluator is intentionally NOT re-exported here.
# Importing it eagerly loads RAGAS (and emits deprecation warnings) in
# non-eval paths. Import it directly where needed:
#   from src.pipeline.evaluation import RAGEvaluator
