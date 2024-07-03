from .base import TemplateDataset


class GenericDataset(TemplateDataset):
    "Generic topic dataset, which creates LLM Contexts and References based on a set of topics."
    name = "generic"
    query_template = (
        "Generate a context and claims about a {subtopic} related to {topic} with difficulty rating {difficulty} in the form of {context_type}"
    )
    params = dict(
        difficulty = [
            "easy",
            "medium", 
            "hard"
        ],
        subtopic=[
            "current event",
            "how-to",
            "historical fiction",
            "non-fiction",
        ],
        topic=[
            "science",
            "politics",
            "parenting",
            "travel",
            "cuisine",
            "sports",
            "pop culture",
            "tech",
            "history",
            "space",
            "economics",
        ],
        context_type = [
            "panel discussion",
            "chat messages",
            "book",
            "monologue",
            "screenplay"
        ],
    )