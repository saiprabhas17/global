"""Pin the requested migration chain and non-destructive historical backfill."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

from sqlalchemy import Table

from app.models.eval import EvalPair, EvalPairLabel


def migration() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "alembic" / "versions" / "gmee02_research_provenance.py"
    spec = importlib.util.spec_from_file_location("research_migration_unit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_revision_and_legacy_backfill_preserve_votes(monkeypatch):
    module = migration()
    assert module.revision == "gmee02" and module.down_revision == "gmee01"
    operation = MagicMock()
    monkeypatch.setattr(module, "op", operation)
    module.upgrade()
    sql = "\n".join(str(call.args[0]) for call in operation.execute.call_args_list)
    assert "row_number() OVER" in sql and "r.rn = 1" in sql
    assert "LEAST(claim_a_id, claim_b_id)" in sql
    assert "DELETE" not in sql.upper()
    added = {(call.args[0], call.args[1].name): call.args[1] for call in operation.add_column.call_args_list}
    assert added[("eval_pair_labels", "origin")].server_default.arg == "legacy"
    assert added[("eval_pairs", "split")].server_default.arg == "unassigned"
    assert added[("eval_pairs", "event_group")].server_default.arg == "unassigned"
    assert added[("eval_pairs", "canonical_key")].nullable
    operation.create_unique_constraint.assert_any_call("uq_eval_pairs_canonical_key", "eval_pairs", ["canonical_key"])
    operation.create_unique_constraint.assert_any_call("uq_eval_label_pair_annotator_origin", "eval_pair_labels", ["pair_id", "annotator", "origin"])


def test_downgrade_refuses_loss_instead_of_purging_history(monkeypatch):
    module = migration()
    operation = MagicMock()
    monkeypatch.setattr(module, "op", operation)
    module.downgrade()
    guard = str(operation.execute.call_args.args[0])
    assert "RAISE EXCEPTION" in guard and "history <> '[]'::jsonb" in guard
    assert "DELETE" not in guard.upper()


def test_model_defaults_and_constraints_are_conservative():
    assert EvalPairLabel.__table__.c.origin.default.arg == "legacy"
    assert EvalPairLabel.__table__.c.origin.server_default.arg == "legacy"
    assert EvalPair.__table__.c.split.default.arg == "unassigned"
    assert EvalPair.__table__.c.canonical_key.nullable
    assert isinstance(EvalPair.__table__, Table)
    assert isinstance(EvalPairLabel.__table__, Table)
    assert "uq_eval_pairs_canonical_key" in {c.name for c in EvalPair.__table__.constraints}
    assert "uq_eval_label_pair_annotator_origin" in {c.name for c in EvalPairLabel.__table__.constraints}
