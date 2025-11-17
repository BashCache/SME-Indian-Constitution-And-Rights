# # utils/guardrails.py
# from langchain.schema.runnable import Runnable
# from typing import Dict, Any
# import re
# import json

# BANNED_KEYWORDS = ["bomb", "attack", "weapon", "password", "api_key", "credit card"]
# MAX_QUERY_LENGTH = 1000


# class GuardrailRunnable(Runnable):
#     """A Runnable that validates user input before passing it to the agent."""

#     def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
#         input_data_ = input_data.get("input", "")

#         # Convert to json and extract the input from it
#         input_data_json = json.loads(input_data_) 
#         query = input_data_json.get("input", "")
#         history = input_data_json.get("history", [])

#         print(f"Query in guardrail: {query}")

#         # 1️⃣ Check for empty input
#         if not query.strip():
#             raise ValueError("❌ Guardrail: Query cannot be empty.")

#         # 2️⃣ Check length
#         if len(query) > MAX_QUERY_LENGTH:
#             raise ValueError(f"❌ Guardrail: Query too long ({len(query)} chars).")

#         # 3️⃣ Banned keywords
#         for word in BANNED_KEYWORDS:
#             if re.search(rf"\b{re.escape(word)}\b", query, re.IGNORECASE):
#                 raise ValueError(f"❌ Guardrail: Disallowed keyword '{word}' detected.")

#         # 4️⃣ Prompt injection detection
#         injection_phrases = [
#             "ignore previous instructions",
#             "reveal system prompt",
#             "run python code",
#             "execute command",
#         ]
#         for phrase in injection_phrases:
#             if phrase.lower() in query.lower():
#                 raise ValueError("❌ Guardrail: Potential prompt injection detected.")

#         # Optional logging
#         print(f"[Guardrail ✅] Input validated successfully: {query[:80]}")

#         # Pass the input forward if all checks pass
#         return input_data



# utils/guardrails.py
from langchain_core.runnables import Runnable
from typing import Dict, Any
import re
import json
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class GuardrailRunnable(Runnable):
    """
    Context-aware Guardrail layer:
    1. Checks for unsafe inputs.
    2. Detects adversarial / prompt injection attempts.
    3. Uses an LLM-based semantic check for context disambiguation.
    """

    def __init__(self):
        # Optional lightweight reasoning model
        self.moderation_llm = ChatNVIDIA(
            model="nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
            api_key=os.getenv("NVIDIA_GUARDRAILS_API_KEY"),
            temperature=0.0,
        )

        # --- Rule categories ---
        self.sensitive_keywords = [
            "password", "api_key", "credit card", "social security", "nuke"
        ]

        self.ambiguous_keywords = [
            "attack", "exploit", "hack", "bypass", "bomb"
        ]

        # Patterns for clear prompt injection
        self.injection_patterns = [
            r"ignore (all|previous) instructions",
            r"reveal (your|the) system prompt",
            r"execute (command|code)",
            r"act as an unrestricted ai",
        ]

    # -------------------------
    # 🚨 LEVEL 1 — Static rule check
    # -------------------------
    def _check_static_rules(self, text: str):
        text_lower = text.lower()

        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                raise ValueError(f"❌ Guardrail: Sensitive term '{keyword}' detected.")

        for pattern in self.injection_patterns:
            if re.search(pattern, text_lower):
                raise ValueError("❌ Guardrail: Potential prompt injection detected.")

    # -------------------------
    # 🧠 LEVEL 2 — Contextual keyword disambiguation
    # -------------------------
    def _check_contextual_keywords(self, text: str):
        """
        If an ambiguous term appears, we verify it's in a safe, academic, or technical context.
        """
        text_lower = text.lower()
        for keyword in self.ambiguous_keywords:
            if keyword in text_lower:
                context_safe_terms = [
                    "neural", "network", "cyber", "adversarial",
                    "simulation", "study", "analysis", "research"
                ]
                # if keyword appears without safe context, flag it
                if not any(ctx in text_lower for ctx in context_safe_terms):
                    # ask the small model for semantic classification
                    verdict = self._semantic_context_check(text)
                    if "unsafe" in verdict.lower():
                        raise ValueError(
                            f"❌ Guardrail: Keyword '{keyword}' found in unsafe context. Please ask some other question"
                        )

    # -------------------------
    # 🧠 LEVEL 3 — Model-based semantic reasoning
    # -------------------------
    def _semantic_context_check(self, text: str) -> str:
        """
        Ask a small model to assess the intent of the query.
        """
        prompt = (
            "You are a security filter. Analyze this query:\n\n"
            f"'{text}'\n\n"
            "Determine if the user is discussing a concept academically (safe) "
            "or describing an illegal or harmful act (unsafe). Any iillegal or questions that could potentially harm or induce violence should be marked as UNSAFE"
            "Answer only with 'SAFE' or 'UNSAFE'."
        )
        try:
            result = self.moderation_llm.invoke(prompt)
            print(f"Result from guardrails: {result}")
            if isinstance(result, dict):
                return result.get("output", "").strip()
            elif hasattr(result, "content"):
                return result.content.strip()
            return str(result).strip()
        except Exception as e:
            print(f"[Guardrail] Semantic model fallback: {e}")
            return "SAFE"

    # -------------------------
    # ✅ ENTRYPOINT
    # -------------------------
    def invoke(self, input_data: Dict[str, Any], config=None) -> Dict[str, Any]:
        input_data_ = input_data.get("input", "")

        # Convert to json and extract the input from it
        input_data_json = json.loads(input_data_) 
        query = input_data_json.get("input", "")

        print(f"Query in guardrail: {query}")

        if not query.strip():
            raise ValueError("❌ Guardrail: Query cannot be empty.")

        if len(query) > 2000:
            raise ValueError("❌ Guardrail: Query too long.")

        # 1️⃣ Static filtering
        self._check_static_rules(query)

        # 2️⃣ Contextual keyword disambiguation
        self._check_contextual_keywords(query)

        verdict = self._semantic_context_check(query)
        if "unsafe" in verdict.lower():
            raise ValueError(
                f"❌ Guardrail: Unsafe query flagged by model. Please ask some other question"
            )

        print(f"[Guardrail ✅] Input passed secure checks.")
        return input_data

