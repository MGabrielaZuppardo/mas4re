import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

# Raised inside langgraph's own import (cache/base); nothing here uses the serializer it names.
# langchain_core re-enables its warnings when imported, so the filter goes in after it.
# Drop this filter when langgraph is upgraded past it.
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects`",
    category=LangChainPendingDeprecationWarning,
)
