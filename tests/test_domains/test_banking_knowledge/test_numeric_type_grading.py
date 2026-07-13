"""Archetype regression tests: DB state must not depend on numeric formatting.

Each test drives the same semantic action through the real tool dispatch path
twice — once with amounts emitted as ints (`33`) and once as floats (`33.00`) —
and asserts the graded DB hashes are identical. These are the four archetypes
of the int/float grader bug:

1. credit card applications  (raw stored value + value-derived record ID)
2. savings/checking credits  (raw stored value + value-derived txn ID)
3. discoverable-call logging (raw stored args dict + json.dumps-derived ID)
4. dispute/CLI records       (raw stored value, stable ID)
"""

import json

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import EvaluationCriteria, RewardType, Task
from tau2.domains.banking_knowledge.data_model import DatabaseTable, TransactionalDB
from tau2.domains.banking_knowledge.tools import KnowledgeTools, KnowledgeUserTools
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator


def make_db() -> TransactionalDB:
    return TransactionalDB(
        users=DatabaseTable(
            data={
                "u1": {
                    "name": "Pat Doe",
                    "user_id": "u1",
                    "address": "1 Main St",
                    "email": "pat@example.com",
                    "phone_number": "555-0100",
                    "date_of_birth": "01/01/1990",
                }
            }
        ),
        accounts=DatabaseTable(
            data={
                "sav_u1_gold": {
                    "account_id": "sav_u1_gold",
                    "user_id": "u1",
                    "class": "savings",
                    "status": "ACTIVE",
                    "current_holdings": "500.00",
                },
                "chk_u1": {
                    "account_id": "chk_u1",
                    "user_id": "u1",
                    "class": "checking",
                    "status": "ACTIVE",
                    "current_holdings": "1000.00",
                },
            }
        ),
        credit_card_accounts=DatabaseTable(
            data={
                "cc_u1_plat": {
                    "account_id": "cc_u1_plat",
                    "user_id": "u1",
                    "card_type": "Platinum Rewards Card",
                    "current_balance": "$100.00",
                }
            }
        ),
    )


def make_env(db: TransactionalDB | None = None, solo_mode: bool = False) -> Environment:
    db = db if db is not None else make_db()
    return Environment(
        domain_name="banking_knowledge",
        policy="",
        tools=KnowledgeTools(db),
        user_tools=KnowledgeUserTools(db),
    )


def assert_db_hashes_equal(env_a: Environment, env_b: Environment) -> None:
    assert env_a.get_db_hash() == env_b.get_db_hash()
    assert env_a.get_user_db_hash() == env_b.get_user_db_hash()


class TestArchetype1CreditCardApplication:
    def test_annual_income_formatting_does_not_change_db(self):
        env_int, env_float = make_env(), make_env()
        for env, income in ((env_int, 85000), (env_float, 85000.00)):
            result = env.make_tool_call(
                tool_name="apply_for_credit_card",
                requestor="user",
                card_type="Platinum Rewards Card",
                customer_name="Pat Doe",
                annual_income=income,
            )
            assert "Error" not in result
        assert_db_hashes_equal(env_int, env_float)


class TestArchetype2SavingsCredit:
    def test_credit_amount_formatting_does_not_change_db(self):
        env_int, env_float = make_env(), make_env()
        for env, amount in ((env_int, 33), (env_float, 33.00)):
            result = env.make_tool_call(
                tool_name="apply_savings_account_credit_6831",
                requestor="assistant",
                account_id="sav_u1_gold",
                amount=amount,
                credit_type="interest_correction",
            )
            assert "Credit applied successfully" in result
        assert_db_hashes_equal(env_int, env_float)


class TestArchetype3DiscoverableCallLogging:
    def test_user_tool_call_args_formatting_does_not_change_db(self):
        env_int, env_float = make_env(), make_env()
        for env, amount_json in ((env_int, "200"), (env_float, "200.00")):
            give = env.make_tool_call(
                tool_name="give_discoverable_user_tool",
                requestor="assistant",
                discoverable_tool_name="deposit_check_3847",
            )
            assert "Error" not in give
            result = env.make_tool_call(
                tool_name="call_discoverable_user_tool",
                requestor="user",
                discoverable_tool_name="deposit_check_3847",
                arguments=json.dumps(
                    {"account_id": "chk_u1", "check_amount": json.loads(amount_json)}
                ),
            )
            assert "Error" not in result
        assert_db_hashes_equal(env_int, env_float)


class TestArchetype4ValueOnlyRecord:
    def test_cli_request_amount_formatting_does_not_change_db(self):
        env_int, env_float = make_env(), make_env()
        for env, amount in ((env_int, 2500), (env_float, 2500.0)):
            result = env.make_tool_call(
                tool_name="submit_credit_limit_increase_request_7392",
                requestor="assistant",
                credit_card_account_id="cc_u1_plat",
                user_id="u1",
                requested_increase_amount=amount,
            )
            assert "Error" not in result
        assert_db_hashes_equal(env_int, env_float)


class TestEvaluatorEndToEnd:
    def test_int_emitting_trajectory_matches_float_authored_gold(self):
        """Gold actions authored `33.00`, trajectory emitted `33` -> db_match."""
        task = Task(
            id="numeric_regression",
            user_scenario={"instructions": "Ask for a $33 interest correction."},
            evaluation_criteria=EvaluationCriteria(
                actions=[
                    {
                        "action_id": "a0",
                        "name": "apply_savings_account_credit_6831",
                        "requestor": "assistant",
                        "arguments": {
                            "account_id": "sav_u1_gold",
                            "amount": 33.00,
                            "credit_type": "interest_correction",
                        },
                    }
                ],
                reward_basis=[RewardType.DB],
            ),
        )
        trajectory = [
            UserMessage(id="1", role="user", content="I'm owed a $33 correction"),
            AssistantMessage(
                id="2",
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="apply_savings_account_credit_6831",
                        arguments={
                            "account_id": "sav_u1_gold",
                            "amount": 33,
                            "credit_type": "interest_correction",
                        },
                    )
                ],
            ),
            ToolMessage(id="t1", role="tool", content="Credit applied successfully!"),
            AssistantMessage(id="3", role="assistant", content="Done."),
        ]
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=make_env,
            task=task,
            full_trajectory=trajectory,
        )
        assert reward_info.db_check is not None
        assert reward_info.db_check.db_match
        assert reward_info.reward == 1.0
