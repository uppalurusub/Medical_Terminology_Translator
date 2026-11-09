import os
import time
from datetime import datetime
from typing import Any

import mlflow
from dotenv import load_dotenv
from langsmith import traceable

from app.config.settings import settings
logger = settings.get_logger()

from app.prompts.translator_prompt import translator_prompt
from app.implementations.translator_impl import MedicalTranslation

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(
    os.getenv("MLFLOW_EXPERIMENT_NAME", "Medical_Terminology_Translator_Interface")
)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def extract_usage(raw_response: Any) -> dict:
    """Extract usage from the actual AIMessage returned by LangChain."""
    usage = _as_dict(getattr(raw_response, "usage_metadata", None))
    metadata = _as_dict(getattr(raw_response, "response_metadata", None))
    token_usage = _as_dict(metadata.get("token_usage"))
    provider_usage = _as_dict(metadata.get("usage"))

    input_tokens = _as_int(
        usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or provider_usage.get("prompt_tokens")
        or provider_usage.get("input_tokens")
    )
    output_tokens = _as_int(
        usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or provider_usage.get("completion_tokens")
        or provider_usage.get("output_tokens")
    )
    total_tokens = _as_int(
        usage.get("total_tokens")
        or token_usage.get("total_tokens")
        or provider_usage.get("total_tokens")
        or input_tokens + output_tokens
    )

    provider_cost = (
        token_usage.get("cost")
        if token_usage.get("cost") is not None
        else provider_usage.get("cost", metadata.get("cost"))
    )

    input_rate = float(os.getenv("INPUT_COST_PER_1M_TOKENS", "0") or 0)
    output_rate = float(os.getenv("OUTPUT_COST_PER_1M_TOKENS", "0") or 0)
    estimated_input_cost = input_tokens * input_rate / 1_000_000
    estimated_output_cost = output_tokens * output_rate / 1_000_000
    estimated_total_cost = estimated_input_cost + estimated_output_cost

    try:
        if provider_cost is not None:
            total_cost = float(provider_cost)
            cost_source = "provider_reported"
        else:
            total_cost = estimated_total_cost
            cost_source = (
                "configured_rate_estimate"
                if input_rate > 0 or output_rate > 0
                else "unavailable_no_rates"
            )
    except (TypeError, ValueError):
        total_cost = estimated_total_cost
        cost_source = "configured_rate_estimate"

    # Unless the provider supplies a reliable split, retain the rate-based split.
    input_cost = estimated_input_cost
    output_cost = estimated_output_cost
    if provider_cost is not None and total_cost == 0 and (input_tokens or output_tokens):
        # Do not imply that a missing split is known when only total cost exists.
        input_cost = 0.0
        output_cost = 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "cost_source": cost_source,
        "model": metadata.get("model_name") or metadata.get("model") or os.getenv("model", ""),
        "provider": "openrouter",
    }


class TranslatorService:

    @staticmethod
    @traceable(
        name="medical_terminology_translation",
        run_type="chain",
        tags=["medical-translator", "openrouter", "structured-output"],
        metadata={"component": "TranslatorService", "feature": "medical_translation"},
    )
    def translate(medical_term: str, language: str = "English") -> MedicalTranslation:
        started = time.perf_counter()
        started_at = datetime.now()
        logger.info("Translation request received")

        try:
            llm = settings.get_model_client()

            # include_raw=True keeps the original AIMessage so usage metadata is
            # available even when the parsed result is a Pydantic object.
            structured_llm = llm.with_structured_output(
                MedicalTranslation,
                include_raw=True,
            )
            chain = translator_prompt | structured_llm

            run_name = f"Medical_Translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("start_time", started_at.strftime("%Y-%m-%d %H:%M:%S"))
                mlflow.log_param("model", os.getenv("model", "unknown"))
                mlflow.log_param("language", language)
                mlflow.set_tag("source", "FastAPI")
                mlflow.set_tag("endpoint", "/translator/translate")
                mlflow.set_tag("provider", "openrouter")
                mlflow.set_tag("feature", "medical_terminology_translation")

                result = chain.invoke(
                    {"medical_term": medical_term, "language": language},
                    config={
                        "run_name": "medical_translation_llm_chain",
                        "tags": [
                            "medical-translator",
                            "openrouter",
                            f"language:{language.lower()}",
                        ],
                        "metadata": {
                            "provider": "openrouter",
                            "model": os.getenv("model", "unknown"),
                            "feature": "medical_terminology_translation",
                        },
                    },
                )

                mlflow.log_param("input", medical_term)
                mlflow.log_param("output", result)

                raw = result.get("raw")
                parsed = result.get("parsed")
                parsing_error = result.get("parsing_error")
                if parsing_error:
                    raise ValueError(f"Structured output parsing failed: {parsing_error}")
                if parsed is None:
                    raise ValueError("LLM returned no parsed MedicalTranslation response")

                usage = extract_usage(raw)
                execution_time = time.perf_counter() - started

                # MLflow metrics must be numeric. Do not log fabricated usage.
                for key in (
                    "input_tokens", "output_tokens", "total_tokens",
                    "input_cost_usd", "output_cost_usd", "total_cost_usd",
                ):
                    mlflow.log_metric(key, float(usage[key]))
                mlflow.log_metric("execution_time_seconds", round(execution_time, 4))
                mlflow.set_tag("usage_source", "raw_ai_message")
                mlflow.set_tag("cost_source", usage["cost_source"])
                mlflow.set_tag("trace_tags", "medical-translator,openrouter,structured-output")

                mlflow.log_text(
                    str({
                        "medical_term": medical_term,
                        "language": language,
                        "translation": parsed.model_dump(),
                        "usage": usage,
                    }),
                    "translation_trace_summary.json",
                )

                logger.info(
                    "LLM usage | input_tokens={} | output_tokens={} | total_tokens={} "
                    "| input_cost_usd={} | output_cost_usd={} | total_cost_usd={} "
                    "| cost_source={} | execution_time_seconds={:.3f}",
                    usage["input_tokens"], usage["output_tokens"], usage["total_tokens"],
                    usage["input_cost_usd"], usage["output_cost_usd"],
                    usage["total_cost_usd"], usage["cost_source"], execution_time,
                )

            logger.success("Medical translation completed")
            return parsed

        except Exception as ex:
            logger.exception("Translation failed: {}", str(ex))
            raise
