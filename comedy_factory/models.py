"""Models the pipeline steps run on."""

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import StructuredOutputMode
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.settings import ModelSettings


class StructuredOutputModel(WrapperModel):
    """A model that asks for a step's structured output natively
    (JSON-schema constrained) where it supports that, as with Gemini and Claude
    Sonnet/Opus, otherwise with the schema in the prompt (e.g. Claude Haiku).
    Tool-call output, pydantic-ai's default, is avoided: Anthropic rejects
    forced tool use while thinking is enabled. Pinned on the model rather than
    chosen per run because the agents' output types have to stay fixed: an
    agent with output validators takes no run-time output type.
    Non-structured outputs pass through as is."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        native = wrapped.profile.get("supports_json_schema_output", False)
        self._output_mode: StructuredOutputMode = "native" if native else "prompted"

    @property
    def profile(self) -> ModelProfile:  # type: ignore[override]
        return merge_profile(
            self.wrapped.profile, ModelProfile(default_structured_output_mode=self._output_mode)
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # The wrapped model resolves an undecided ("auto") output mode from its
        # own profile, so decide it here first.
        return await self.wrapped.request(
            messages,
            model_settings,
            model_request_parameters.with_default_output_mode(self._output_mode),
        )
