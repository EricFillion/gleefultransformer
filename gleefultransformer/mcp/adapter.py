import json
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .errors import ServiceError
from .service import RunService

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Cursor = Annotated[str, Field(max_length=512)]
PageLimit = Annotated[int, Field(ge=1, le=100)]
SequenceLimit = Annotated[int, Field(ge=2, le=32768)]
Operation = Literal["finetune", "evaluation", "reply"]
Role = Literal["train", "validation", "test"]
State = Literal[
    "queued", "running", "cancel_requested", "succeeded", "failed", "cancelled", "interrupted"
]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    model_id: Identifier
    train_dataset_id: Identifier
    max_steps: Annotated[int, Field(ge=1)]
    validation_dataset_id: Identifier | None = None
    max_sequence_tokens: SequenceLimit | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    role: Annotated[str, Field(min_length=1, max_length=64)]
    content: Annotated[str, Field(min_length=1, max_length=16384)]


class ModelProfile(BaseModel):
    model_id: str
    revision: str
    origin: str
    capabilities: list[str]
    tokenizer_revision: str | None
    max_sequence_tokens: int | None


class DatasetSplit(BaseModel):
    dataset_id: str
    revision: str | None
    display_name: str
    role: Role | None
    format: Literal["chat_jsonl"]
    row_count: int | None


class ModelCatalog(BaseModel):
    items: list[ModelProfile]


class DatasetCatalog(BaseModel):
    items: list[DatasetSplit]


class DatasetCheck(BaseModel):
    dataset_id: str
    model_id: str
    valid: bool
    row_count: int | None = None
    sha256: str | None = None
    bytes: int | None = None
    min_tokens: int | None = None
    max_tokens: int | None = None
    total_tokens: int | None = None
    over_limit: int | None = None
    max_sequence_tokens: int | None = None
    issues: list[dict[str, JsonValue]]


class Progress(BaseModel):
    current_step: int | None = None
    total_steps: int | None = None


class RunError(BaseModel):
    code: str
    message: str
    retryable: bool


class EvaluationMetrics(BaseModel):
    tokenizer_revision: str | None
    dataset_id: str
    dataset_revision: str
    max_sequence_tokens: int
    row_count: int
    loss: float | None
    step: int | None = None
    warnings: list[str] | None = None


class EvaluationReport(EvaluationMetrics):
    model_id: str
    model_revision: str


class TrainingValidationReport(EvaluationMetrics):
    input_model_id: str
    input_model_revision: str


class FineTuneResult(BaseModel):
    final_model_id: str
    final_train_loss: float | None
    best_eval_loss: float | None
    validation: TrainingValidationReport | None = None


class Reply(BaseModel):
    text: str
    model_revision: str


class RunSnapshot(BaseModel):
    run_id: str
    kind: Operation
    state: State
    phase: str
    model_id: str | None
    published_model_id: str | None
    created_at: str
    started_at: str | None
    updated_at: str
    cancel_requested_at: str | None
    finished_at: str | None
    progress: Progress
    metrics: dict[str, JsonValue]
    artifacts: dict[str, JsonValue]
    result: FineTuneResult | EvaluationReport | Reply | None
    error: RunError | None
    poll_after_ms: int | None
    stop_action: Literal["cancelled_queued", "requested", "already_requested", "already_terminal"] | None = None


class RunPage(BaseModel):
    items: list[RunSnapshot]
    next_cursor: str | None


def _result(result_type, operation, *args, **kwargs):
    try:
        data = operation(*args, **kwargs)
    except ServiceError as exc:
        raise ToolError(json.dumps(exc.to_dict(), ensure_ascii=True)) from exc
    return result_type.model_validate(data)


def create_server(service: RunService) -> MCPServer:
    server = MCPServer(
        "GleefulTransformer",
        title="Gleeful Transformer",
        instructions=(
            "Choose registered model and dataset IDs. Submit supervised chat fine-tuning, "
            "then poll by run ID. Evaluate the exported model on the same held-out split before comparing it."
        ),
    )

    @server.tool()
    def gleeful_find_model_profiles() -> ModelCatalog:
        """List registered and exported chat models and their revisions."""
        return _result(ModelCatalog, service.find_models)

    @server.tool()
    def gleeful_find_dataset_splits() -> DatasetCatalog:
        """List registered chat JSONL splits and their roles."""
        return _result(DatasetCatalog, service.find_datasets)

    @server.tool()
    def gleeful_check_dataset_contract(
        dataset_id: Identifier, model_id: Identifier, max_sequence_tokens: SequenceLimit | None = None,
    ) -> DatasetCheck:
        """Validate a registered split against a pinned model's chat template without loading weights."""
        return _result(DatasetCheck, service.check_dataset, dataset_id, model_id, max_sequence_tokens)

    @server.tool()
    def gleeful_submit_finetune(recipe: Recipe, request_id: Identifier) -> RunSnapshot:
        """Validate and enqueue a supervised fine-tune. Retry with the same request ID."""
        return _result(
            RunSnapshot, service.submit_finetune, recipe.model_dump(exclude_none=True),
            request_id,
        )

    @server.tool()
    def gleeful_submit_dataset_evaluation(
        model_id: Identifier, dataset_id: Identifier, request_id: Identifier,
        max_sequence_tokens: SequenceLimit | None = None,
    ) -> RunSnapshot:
        """Enqueue loss evaluation on a registered chat split."""
        return _result(
            RunSnapshot, service.submit_evaluation, model_id, dataset_id, request_id,
            max_sequence_tokens=max_sequence_tokens,
        )

    @server.tool()
    def gleeful_read_run_snapshot(run_id: Identifier) -> RunSnapshot:
        """Read the latest persisted run state, progress, metrics, and ready artifacts."""
        return _result(RunSnapshot, service.read_run, run_id)

    @server.tool()
    def gleeful_find_experiment_runs(
        kind: Operation | None = None, state: State | None = None, model_id: Identifier | None = None,
        cursor: Cursor | None = None, limit: PageLimit = 20,
    ) -> RunPage:
        """Find recent workspace runs, including terminal failures and published model IDs."""
        return _result(RunPage, service.find_runs, kind, state, model_id, cursor, limit)

    @server.tool()
    def gleeful_request_run_stop(run_id: Identifier) -> RunSnapshot:
        """Cancel a queued run or request a cooperative stop for an active run."""
        return _result(RunSnapshot, service.request_stop, run_id)

    @server.tool()
    def gleeful_reply_to_messages(
        model_id: Identifier, messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=64)],
        max_new_tokens: Annotated[int, Field(ge=1, le=512)],
    ) -> Reply:
        """Return one bounded reply from a registered or exported model using the supplied history."""
        return _result(
            Reply, service.reply, model_id, [message.model_dump() for message in messages], max_new_tokens
        )

    return server
