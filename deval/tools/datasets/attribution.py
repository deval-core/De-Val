from .base import TemplateDataset


class AttributionDataset(TemplateDataset):
    "dataset for attribution task, which creates LLM Topics and context types for evaluating attribution."
    name = "attribution"
    query_template = (
        "Generate a context and action items about a {topic} in the form of {context_type}"
    )
    params = dict(
        topic=[
            "credit card",
            "clothes",
            "wedding venues",
            "mobile app",
            "Software-as-a-service",
            "toys",
            "tech gadgets"
        ],
        context_type = [
            "chat messages",
            "customer service call",
            "technical support chat",
            "sales call",
            "meeting transcript",
            "technical brainstorm"
        ],
    )