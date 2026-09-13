from types import SimpleNamespace

import pytest

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.task_suite.load_suites import get_suite
from ipi_aware.data_collection import agentdojo_collector


class DummyPipeline(BasePipelineElement):
    name = "gpt-4o-2024-05-13"

    def query(self, query, runtime, env, messages=[], extra_args={}):
        raise NotImplementedError


def test_vendored_agentdojo_registers_long_horizon_attack():
    suite = get_suite("v1.2.2", "workspace")
    attack = load_attack("long_horizon_important_instructions", suite, DummyPipeline())

    user_task = suite.user_tasks["user_task_39"]
    injection_task = suite.injection_tasks["injection_task_0"]
    injections = attack.attack(user_task, injection_task)

    assert list(injections) == ["email_password_reset_injection"]
    assert injection_task.GOAL in injections["email_password_reset_injection"]


def test_vendored_agentdojo_registers_long_horizon_tool_knowledge_attack():
    suite = get_suite("v1.2.2", "workspace")
    attack = load_attack("long_horizon_tool_knowledge", suite, DummyPipeline())

    user_task = suite.user_tasks["user_task_39"]
    injection_task = suite.injection_tasks["injection_task_0"]
    injections = attack.attack(user_task, injection_task)

    assert list(injections) == ["email_password_reset_injection"]
    assert injection_task.GOAL in injections["email_password_reset_injection"]


def test_vendored_agentdojo_registers_persistent_long_horizon_attack():
    suite = get_suite("v1.2.2", "workspace")
    attack = load_attack("persistent_long_horizon_important_instructions", suite, DummyPipeline())

    user_task = suite.user_tasks["user_task_38"]
    injection_task = suite.injection_tasks["injection_task_0"]
    injections = attack.attack(user_task, injection_task)

    assert list(injections) == ["drive_recipe_injection"]
    assert injection_task.GOAL in injections["drive_recipe_injection"]


def test_vendored_agentdojo_registers_persistent_long_horizon_tool_knowledge_attack():
    suite = get_suite("v1.2.2", "workspace")
    attack = load_attack("persistent_long_horizon_tool_knowledge", suite, DummyPipeline())

    user_task = suite.user_tasks["user_task_38"]
    injection_task = suite.injection_tasks["injection_task_0"]
    injections = attack.attack(user_task, injection_task)

    assert list(injections) == ["drive_recipe_injection"]
    assert injection_task.GOAL in injections["drive_recipe_injection"]


def test_collector_rejects_preloaded_upstream_agentdojo(monkeypatch, tmp_path):
    upstream = tmp_path / "site-packages" / "agentdojo" / "__init__.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("__version__ = '0.1.35'\n")
    monkeypatch.setitem(
        __import__("sys").modules,
        "agentdojo",
        SimpleNamespace(__file__=str(upstream)),
    )

    with pytest.raises(RuntimeError, match="non-vendored agentdojo"):
        agentdojo_collector._ensure_vendor_agentdojo_path()
