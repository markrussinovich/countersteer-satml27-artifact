import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import NewType

from rich.progress import track

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import get_text_content_as_str

# user_task_id -> injection_task_id -> vector_id -> value
Injections = NewType("Injections", dict[str, dict[str, dict[str, str]]])

DEFAULT_MODEL_NAME = "the AI language model"
"""Default name of the victim model."""
DEFAULT_USER_NAME = "the user"
"""Default name of the victim user."""


@dataclass(frozen=True)
class InjectionOccurrence:
    """A single observed canary placeholder occurrence in the ground-truth trajectory."""

    vector_id: str
    message_index: int
    start_index: int
    end_index: int


class BaseAttack(abc.ABC):
    """Base class for attacks.

    Args:
        task_suite: The task suite with the user and injection tasks being used for the benchmark.
        target_pipeline: The target pipeline that is attacked.

    """

    is_dos_attack: bool = False
    """Whether the attack is a DOS attack. If `True`, then the benchmarking
    will be done with only one injection task."""

    name: str
    """The name of the attack."""

    def __init__(self, task_suite: TaskSuite, target_pipeline: BasePipelineElement) -> None:
        self.task_suite = task_suite
        self.target_pipeline = target_pipeline
        injection_defaults = task_suite.get_injection_vector_defaults()
        self.canary_injections = {inj: f"<CANARY---{inj}--->" for inj in injection_defaults}
        # TODO: add cache directory to save the attacks that are expensive to compute

    @lru_cache
    def get_injection_candidates(self, user_task: BaseUserTask) -> list[str]:
        """ "Given a user task ID, it returns the list of keys of the injection placeholders
        that are seen by the LLM when executing the task correctly."""
        gt_pipeline = GroundTruthPipeline(user_task)
        runtime = FunctionsRuntime(self.task_suite.tools)
        environment = self.task_suite.load_and_inject_default_environment(self.canary_injections)
        _, _, _, responses, _ = gt_pipeline.query(user_task.PROMPT, runtime, environment)

        injection_candidates = []
        for inj in self.canary_injections:
            found = False
            for response in responses:
                if response["content"] is not None and self.canary_injections[inj] in get_text_content_as_str(
                    response["content"]
                ):
                    found = True
            if found:
                injection_candidates.append(inj)
        if len(injection_candidates) == 0:
            raise ValueError(f"{user_task.ID} is not injectable.")
        return injection_candidates

    def generate_injections(self) -> Injections:
        is_suite_valid, _ = self.task_suite.check()
        if not is_suite_valid:
            raise ValueError(f"Suite {self.task_suite.name} failed checks. Fix the suite first.")

        injections = defaultdict(dict)
        for user_task_id, user_task in track(self.task_suite.user_tasks.items()):
            for inj_task_id, inj_task in self.task_suite.injection_tasks.items():
                attack = self.attack(user_task, inj_task)
                injections[user_task_id][inj_task_id] = attack

        return Injections(dict(injections))

    @lru_cache
    def get_injection_occurrences(self, user_task: BaseUserTask) -> tuple[InjectionOccurrence, ...]:
        """Returns ordered canary placeholder occurrences observed in the ground-truth trajectory."""

        gt_pipeline = GroundTruthPipeline(user_task)
        runtime = FunctionsRuntime(self.task_suite.tools)
        environment = self.task_suite.load_and_inject_default_environment(self.canary_injections)
        _, _, _, responses, _ = gt_pipeline.query(user_task.PROMPT, runtime, environment)

        occurrences: list[InjectionOccurrence] = []
        for message_index, response in enumerate(responses):
            content = response.get("content")
            if content is None:
                continue
            text = get_text_content_as_str(content)
            if not text:
                continue
            for inj, canary in self.canary_injections.items():
                search_start = 0
                while True:
                    start = text.find(canary, search_start)
                    if start < 0:
                        break
                    end = start + len(canary)
                    occurrences.append(
                        InjectionOccurrence(
                            vector_id=inj,
                            message_index=message_index,
                            start_index=start,
                            end_index=end,
                        )
                    )
                    search_start = end

        ordered = tuple(
            sorted(
                occurrences,
                key=lambda occurrence: (
                    occurrence.message_index,
                    occurrence.start_index,
                    occurrence.end_index,
                    occurrence.vector_id,
                ),
            )
        )
        if len(ordered) == 0:
            raise ValueError(f"{user_task.ID} is not injectable.")
        return ordered

    @lru_cache
    def get_latest_injection_candidates(self, user_task: BaseUserTask) -> tuple[str, ...]:
        """Returns the placeholder(s) that appear latest in the ground-truth trajectory."""

        occurrences = self.get_injection_occurrences(user_task)
        latest_occurrence = max(
            occurrences,
            key=lambda occurrence: (
                occurrence.message_index,
                occurrence.start_index,
                occurrence.end_index,
            ),
        )
        latest_candidates = tuple(
            occurrence.vector_id
            for occurrence in occurrences
            if (
                occurrence.message_index,
                occurrence.start_index,
                occurrence.end_index,
            )
            == (
                latest_occurrence.message_index,
                latest_occurrence.start_index,
                latest_occurrence.end_index,
            )
        )
        return latest_candidates

    @lru_cache
    def get_persistent_injection_candidates(self, user_task: BaseUserTask) -> tuple[str, ...]:
        """Returns placeholders that recur across multiple message turns."""

        occurrences = self.get_injection_occurrences(user_task)
        messages_by_vector: dict[str, set[int]] = defaultdict(set)
        for occurrence in occurrences:
            messages_by_vector[occurrence.vector_id].add(occurrence.message_index)

        persistent_candidates: list[str] = []
        seen = set()
        for occurrence in occurrences:
            vector_id = occurrence.vector_id
            if vector_id in seen:
                continue
            if len(messages_by_vector[vector_id]) >= 2:
                seen.add(vector_id)
                persistent_candidates.append(vector_id)
        return tuple(persistent_candidates)

    @lru_cache
    def get_latest_persistent_injection_candidates(self, user_task: BaseUserTask) -> tuple[str, ...]:
        """Returns the latest placeholder(s) among those that recur across message turns."""

        persistent_candidates = set(self.get_persistent_injection_candidates(user_task))
        if len(persistent_candidates) == 0:
            return ()

        occurrences = self.get_injection_occurrences(user_task)
        latest_occurrence_by_vector: dict[str, InjectionOccurrence] = {}
        for occurrence in occurrences:
            if occurrence.vector_id not in persistent_candidates:
                continue
            current = latest_occurrence_by_vector.get(occurrence.vector_id)
            if current is None or (
                occurrence.message_index,
                occurrence.start_index,
                occurrence.end_index,
            ) > (
                current.message_index,
                current.start_index,
                current.end_index,
            ):
                latest_occurrence_by_vector[occurrence.vector_id] = occurrence

        latest_occurrence = max(
            latest_occurrence_by_vector.values(),
            key=lambda occurrence: (
                occurrence.message_index,
                occurrence.start_index,
                occurrence.end_index,
            ),
        )
        latest_candidates = tuple(
            vector_id
            for vector_id, occurrence in latest_occurrence_by_vector.items()
            if (
                occurrence.message_index,
                occurrence.start_index,
                occurrence.end_index,
            )
            == (
                latest_occurrence.message_index,
                latest_occurrence.start_index,
                latest_occurrence.end_index,
            )
        )
        return tuple(sorted(latest_candidates))

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        """Given a user task and an injection task, it returns a dictionary containing the
        injection placeholders that are seen by the agent when execuring the task correctly, and
        the corresponsing injections.

        Args:
            user_task: The user task.
            injection_task: The injection task. It contains the [`GOAL`][agentdojo.base_tasks.BaseInjectionTask.GOAL]
                attribute, that should be used by the attack to instruct the model to execute the injection task.

        """

        ...


class FixedJailbreakAttack(BaseAttack):
    """Attack which uses a fixed template (e.g., `f"ignore previous instructions, do {injection_task.GOAL} instead"`).
    The attack method will substitute {goal}, {user}, {model}, and {task} in the jailbreak string with the given values.
    Subclasses can override `summarize_task` to provide a human-readable summary of the task."""

    user_name: str = DEFAULT_USER_NAME
    """The name of the user. This will be substituted in the jailbreak string to the `{user}` placeholder if present in the string."""
    model_name: str = DEFAULT_MODEL_NAME
    """The name of the model. This will be substituted in the jailbreak string to the `{model}` placeholder if present in the string."""

    def __init__(self, jailbreak: str, task_suite: TaskSuite, target_pipeline: BasePipelineElement) -> None:
        super().__init__(task_suite, target_pipeline)
        self.jailbreak = jailbreak

    def summarize_task(self, task: BaseUserTask) -> str:
        return task.PROMPT

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        injections = {}
        for inj in self.get_injection_candidates(user_task):
            injections[inj] = self.jailbreak.format(
                goal=injection_task.GOAL,
                user=self.user_name,
                model=self.model_name,
                task=self.summarize_task(user_task),
            )
        return injections


def get_model_name_from_pipeline(pipeline: BasePipelineElement) -> str:
    """Gets the model name (in prose) from the pipeline name. Useful to directly address a model to
    make the attack more effective. E.g., if the model in the pipeline is `"gpt-4o-2024-05-13"`,
    then the function will return `"GPT-4"`.

    Args:
        pipeline: The pipeline from which to extract the name.

    """

    if pipeline.name is None:
        raise ValueError("Pipeline name is `None`")

    for full_name, model_name in MODEL_NAMES.items():
        if full_name in pipeline.name:
            return model_name

    raise ValueError(
        f"No valid model name not found in pipeline name `{pipeline.name}`. Valid model names are {sorted(set(MODEL_NAMES.values()))}"
    )
