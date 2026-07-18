from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestExternalAdapterOptions(unittest.TestCase):
    def test_mrag_flagembedding_compat_translates_dtype_once(self) -> None:
        mod = _load_script("query_mrag_reference.py")
        calls = []

        class AutoModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls.append((args, kwargs))
                return "model"

        transformers = SimpleNamespace(AutoModel=AutoModel)
        with patch.dict("sys.modules", {"transformers": transformers}):
            mod._install_flagembedding_transformers_compat()
            mod._install_flagembedding_transformers_compat()

        result = AutoModel.from_pretrained("model-id", dtype="float32", revision="main")

        self.assertEqual(result, "model")
        self.assertEqual(calls, [(("model-id",), {"torch_dtype": "float32", "revision": "main"})])

    def test_mrag_dependency_check_is_mode_specific(self) -> None:
        mod = _load_script("query_mrag_reference.py")
        available = ({*mod.REQUIRED_MODULES, "FlagEmbedding"} - {"networkx"})

        def find_spec(name):
            return object() if name in available else None

        args = argparse.Namespace(python=Path("missing-python"), mode="dense")
        with patch.object(mod.importlib.util, "find_spec", side_effect=find_spec):
            dense = mod._dependency_report(args)
        self.assertTrue(dense["runnable"])
        self.assertEqual(dense["missing_alternative_groups"], [])

        args.mode = "full"
        with patch.object(mod.importlib.util, "find_spec", side_effect=find_spec):
            full = mod._dependency_report(args)
        self.assertFalse(full["runnable"])
        self.assertEqual(
            {group["group"] for group in full["missing_alternative_groups"]},
            {"graph", "reranking", "visual_embedding"},
        )

    def test_local_openai_adapter_checks_require_reachable_endpoint(self) -> None:
        down = {
            "checked": True,
            "url": "http://localhost:8000/v1/models",
            "reachable": False,
            "authorized": None,
            "usable": False,
            "status_code": None,
            "error": "URLError",
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = []

            lightrag = _load_script("query_lightrag_index.py")
            lightrag_repo = root / "lightrag"
            lightrag_repo.mkdir()
            lightrag_index = root / "lightrag-index"
            lightrag_index.mkdir()
            lightrag_corpus = root / "corpus.txt"
            lightrag_corpus.write_text("manual corpus", encoding="utf-8")
            (lightrag_index / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
            lightrag_args = argparse.Namespace(
                repo=lightrag_repo,
                working_dir=lightrag_index,
                corpus=lightrag_corpus,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url="http://localhost:8000/v1",
            )
            lightrag.publish_completion_marker(
                lightrag_index / lightrag.INDEX_SENTINEL,
                lightrag._index_identity(lightrag_args),
                index_files=lightrag._index_files(lightrag_index),
            )
            cases.append((lightrag, lightrag_args))

            raganything = _load_script("query_raganything_index.py")
            raganything_repo = root / "raganything"
            raganything_repo.mkdir()
            raganything_index = root / "raganything-index"
            raganything_index.mkdir()
            raganything_content = root / "content.json"
            raganything_content.write_text("[]", encoding="utf-8")
            (raganything_index / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
            raganything_args = argparse.Namespace(
                repo=raganything_repo,
                lightrag_repo=lightrag_repo,
                working_dir=raganything_index,
                content_list=raganything_content,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url="http://localhost:8000/v1",
            )
            raganything.publish_completion_marker(
                raganything_index / raganything.INDEX_SENTINEL,
                raganything._index_identity(raganything_args),
                index_files=raganything._index_files(raganything_index),
            )
            cases.append((raganything, raganything_args))

            paperqa = _load_script("query_paperqa_index.py")
            paperqa_repo = root / "paperqa"
            paperqa_repo.mkdir()
            paperqa_index = root / "docs.pkl"
            paperqa_index.write_bytes(b"pickle")
            paperqa_chunks = root / "chunks.jsonl"
            paperqa_chunks.write_text('{"text":"manual chunk"}\n', encoding="utf-8")
            paperqa_args = argparse.Namespace(
                repo=paperqa_repo,
                index=paperqa_index,
                chunks=paperqa_chunks,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url="http://localhost:8000/v1",
            )
            paperqa.publish_completion_marker(
                paperqa._index_sentinel(paperqa_index),
                paperqa._index_identity(paperqa_args),
                index=paperqa.file_identity(paperqa_index),
            )
            cases.append((paperqa, paperqa_args))

            for mod, args in cases:
                with self.subTest(adapter=mod.__name__):
                    with (
                        patch.object(mod, "_import_errors", return_value={}),
                        patch.object(mod, "probe_openai_endpoint", return_value=down),
                        patch.dict(os.environ, {}, clear=True),
                    ):
                        report = mod._dependency_report(args)
                    self.assertTrue(report["environment_ready"])
                    self.assertTrue(report["index_ready"])
                    self.assertFalse(report["endpoint_reachable"])
                    self.assertFalse(report["model_service_ready"])
                    self.assertFalse(report["runnable"])

    def test_lightrag_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_lightrag_check_accepts_custom_corpus(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        with patch.object(
            sys,
            "argv",
            ["query_lightrag_index.py", "check", "--corpus", "custom.txt"],
        ):
            args = mod._parse_args()

        self.assertEqual(args.corpus, Path("custom.txt"))

    def test_lightrag_query_fails_closed_before_initialization(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        args = argparse.Namespace(command="query", repo=Path("/tmp/lightrag"))
        report = {"runnable": False, "index_ready": False}
        with (
            patch.object(mod, "_add_repo"),
            patch.object(mod, "_dependency_report", return_value=report),
            patch.object(mod, "_make_rag") as make_rag,
            patch("builtins.print"),
        ):
            self.assertEqual(asyncio.run(mod._main(args)), 2)
        make_rag.assert_not_called()

    def test_lightrag_corpus_id_targets_canonical_status_record(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        self.assertEqual(
            mod._corpus_doc_id("corpus"),
            "doc-9a91380374d05f93bd6ab9362deaec79",
        )

    def test_raganything_allows_dummy_local_key(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(mod._api_key(args), "local")

    def test_paperqa_sets_dummy_local_key(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        args = argparse.Namespace(api_key_env="OPENAI_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._ensure_api_key(args)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local")

    def test_graphrag_applies_dummy_local_key_to_subprocess_env(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            env = mod._env(repo)
        args = argparse.Namespace(api_key_env="GRAPHRAG_API_KEY", allow_missing_api_key=True)
        with patch.dict(os.environ, {}, clear=True):
            mod._apply_local_api_key(args, env)
            self.assertEqual(env["GRAPHRAG_API_KEY"], "local")

    def test_graphrag_reuses_openai_key_by_default(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            env = mod._env(repo)
        args = argparse.Namespace(api_key_env="GRAPHRAG_API_KEY", allow_missing_api_key=False)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            mod._apply_local_api_key(args, env)

        self.assertEqual(env["GRAPHRAG_API_KEY"], "openai-key")

    def test_graphrag_configures_local_api_base_for_completion_and_embedding(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            settings = Path(td) / "settings.yaml"
            settings.write_text(
                """
completion_models:
  default_completion_model:
    model_provider: openai
    model: answer
embedding_models:
  default_embedding_model:
    model_provider: openai
    model: embed
extract_graph:
  entity_types: [organization, person, geo, event]
extract_claims:
  enabled: false
community_reports:
  max_length: 2000
""".lstrip(),
                encoding="utf-8",
            )

            with patch("builtins.print"):
                code = mod._configure_api_base(
                    settings,
                    "http://localhost:8000/v1",
                    embedding_base_url="http://localhost:8001/v1",
                    reasoning_effort="none",
                    llm_max_tokens=2048,
                    entity_types=["organization", "traffic_control_device", "concept"],
                    max_gleanings=0,
                    entity_extraction_max_tokens=4096,
                    entity_extraction_temperature=0.0,
                    entity_extraction_frequency_penalty=0.2,
                    community_report_max_length=300,
                    community_report_max_tokens=768,
                    community_report_temperature=0.0,
                )
            import yaml

            payload = yaml.safe_load(settings.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(
            payload["completion_models"]["default_completion_model"]["api_base"],
            "http://localhost:8000/v1",
        )
        self.assertEqual(
            payload["embedding_models"]["default_embedding_model"]["api_base"],
            "http://localhost:8001/v1",
        )
        self.assertEqual(
            payload["completion_models"]["default_completion_model"]["call_args"],
            {"reasoning_effort": "none", "max_tokens": 2048},
        )
        self.assertNotIn(
            "call_args",
            payload["embedding_models"]["default_embedding_model"],
        )
        self.assertEqual(
            payload["extract_graph"]["entity_types"],
            ["organization", "traffic_control_device", "concept"],
        )
        self.assertEqual(payload["extract_graph"]["max_gleanings"], 0)
        self.assertEqual(
            payload["extract_graph"]["completion_model_id"],
            "entity_extraction_completion_model",
        )
        self.assertEqual(
            payload["extract_claims"]["completion_model_id"],
            "entity_extraction_completion_model",
        )
        self.assertEqual(
            payload["completion_models"]["entity_extraction_completion_model"][
                "call_args"
            ],
            {
                "reasoning_effort": "none",
                "max_tokens": 4096,
                "temperature": 0.0,
                "frequency_penalty": 0.2,
            },
        )
        self.assertEqual(payload["community_reports"]["max_length"], 300)
        self.assertEqual(
            payload["community_reports"]["completion_model_id"],
            "community_report_completion_model",
        )
        self.assertEqual(
            payload["completion_models"]["community_report_completion_model"]["call_args"],
            {"reasoning_effort": "none", "max_tokens": 768, "temperature": 0.0},
        )
        self.assertTrue(
            payload["extract_graph"]["model_instance_name"].startswith("extract_graph_")
        )
        self.assertTrue(
            payload["community_reports"]["model_instance_name"].startswith(
                "community_reports_"
            )
        )
        self.assertNotEqual(
            payload["extract_graph"]["model_instance_name"],
            payload["community_reports"]["model_instance_name"],
        )

    def test_graphrag_removes_upstream_community_report_examples(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            prompts = working_dir / "prompts"
            prompts.mkdir()
            template = (
                "Instructions with {max_report_length}.\n\n"
                "- DETAILED FINDINGS: A list of 5-10 verbose findings.\n\n"
                "# Example Input\n"
                "Enron example that a small model may copy.\n\n"
                "# Real Data\n"
                "Text:\n{input_text}\n\nOutput:"
            )
            for name in mod.COMMUNITY_PROMPT_NAMES:
                (prompts / name).write_text(template, encoding="utf-8")

            with patch("builtins.print"):
                code = mod._sanitize_community_prompts(working_dir)

            rendered = [(prompts / name).read_text(encoding="utf-8") for name in mod.COMMUNITY_PROMPT_NAMES]

        self.assertEqual(code, 0)
        for prompt in rendered:
            self.assertNotIn("# Example Input", prompt)
            self.assertNotIn("Enron", prompt)
            self.assertIn("# Real Data", prompt)
            self.assertIn("{max_report_length}", prompt)
            self.assertIn("{input_text}", prompt)
            self.assertIn("2-4 distinct key insights", prompt)
            self.assertIn("Do not repeat or restate a finding", prompt)
            self.assertNotIn("5-10 verbose findings", prompt)

    def test_graphrag_removes_upstream_extraction_examples(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            prompts = working_dir / "prompts"
            prompts.mkdir()
            template = (
                "Instructions.\n\n"
                "######################\n"
                "-Examples-\n"
                "Synthetic example that a small model may copy.\n\n"
                "######################\n"
                "-Real Data-\n"
                "Text: {input_text}\nOutput:"
            )
            for name in mod.EXTRACTION_PROMPT_NAMES:
                (prompts / name).write_text(template, encoding="utf-8")

            with patch("builtins.print"):
                code = mod._sanitize_extraction_prompts(working_dir)

            rendered = [
                (prompts / name).read_text(encoding="utf-8")
                for name in mod.EXTRACTION_PROMPT_NAMES
            ]

        self.assertEqual(code, 0)
        for prompt in rendered:
            self.assertNotIn("-Examples-", prompt)
            self.assertNotIn("Synthetic example", prompt)
            self.assertIn("-MUTCD Format Example-", prompt)
            self.assertIn("Never repeat or renumber", prompt)
            self.assertIn("<|COMPLETE|>", prompt)
            self.assertIn("-Real Data-", prompt)
            self.assertIn("{input_text}", prompt)

    def test_graphrag_cache_partition_tracks_model_profile(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        base = {
            "model_provider": "openai",
            "model": "small-model",
            "api_base": "http://localhost:8000/v1",
            "api_key": "not-part-of-cache-identity",
            "call_args": {"max_tokens": 512, "temperature": 0},
        }
        changed = copy.deepcopy(base)
        changed["call_args"]["max_tokens"] = 768

        self.assertEqual(
            mod._model_cache_partition("community_reports", base),
            mod._model_cache_partition("community_reports", copy.deepcopy(base)),
        )
        self.assertNotEqual(
            mod._model_cache_partition("community_reports", base),
            mod._model_cache_partition("community_reports", changed),
        )

    def test_graphrag_community_report_levels_are_parsed_and_forwarded(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        self.assertEqual(mod._parse_community_levels("2, 0,2"), (0, 2))
        self.assertIsNone(mod._parse_community_levels("all"))
        with self.assertRaises(argparse.ArgumentTypeError):
            mod._parse_community_levels("-1")

        args = argparse.Namespace(
            python="graph-python",
            allow_missing_api_key=True,
            community_report_token_floor=4096,
        )
        completed = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(mod.subprocess, "run", return_value=completed) as run:
            mod._graphrag_subprocess(
                args,
                {"PYTHONPATH": "paths"},
                ["index", "--root", "workspace"],
                community_levels=(2,),
            )

        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["graph-python", "-c"])
        self.assertIn("install_community_report_level_filter", command[2])
        self.assertIn("install_community_report_token_floor", command[2])
        self.assertEqual(
            json.loads(command[3]),
            {"community_levels": [2], "community_report_token_floor": 4096},
        )
        self.assertEqual(command[4:], ["index", "--root", "workspace"])

    def test_graphrag_community_report_token_floor_preserves_cache_args(self) -> None:
        _load_script("query_graphrag_index.py")
        from gems_rag.graphrag_indexing import _community_report_provider_args
        from pydantic import BaseModel

        class FindingModel(BaseModel):
            summary: str
            explanation: str

        class CommunityReportResponse(BaseModel):
            title: str
            summary: str
            findings: list[FindingModel]
            rating: float
            rating_explanation: str

        cached_args = {
            "max_tokens": 768,
            "response_format": CommunityReportResponse,
            "messages": "report prompt",
        }
        adjusted = _community_report_provider_args(cached_args, 4096)

        self.assertEqual(cached_args["max_tokens"], 768)
        self.assertEqual(adjusted["max_tokens"], 4096)
        self.assertEqual(adjusted["messages"], "report prompt")
        self.assertIsNot(adjusted, cached_args)
        provider_schema = adjusted["response_format"].model_json_schema()
        self.assertEqual(provider_schema["properties"]["findings"]["minItems"], 1)
        self.assertEqual(provider_schema["properties"]["findings"]["maxItems"], 4)
        self.assertEqual(
            provider_schema["$defs"]["FindingModel"]["properties"]["explanation"][
                "maxLength"
            ],
            1000,
        )

        unrelated = {"max_tokens": 768, "response_format": dict}
        self.assertIs(
            _community_report_provider_args(unrelated, 4096),
            unrelated,
        )

    def test_graphrag_query_rejects_unbuilt_community_level(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        query = argparse.Namespace(command="query", method="local", community_level=2)
        marker = {"community_levels": [2]}

        self.assertTrue(mod._query_community_level_available(query, marker))
        query.community_level = 1
        self.assertFalse(mod._query_community_level_available(query, marker))
        query.method = "basic"
        self.assertTrue(mod._query_community_level_available(query, marker))
        self.assertTrue(mod._query_community_level_available(query, {}))

        check = argparse.Namespace(command="check", community_level=2)
        self.assertFalse(
            mod._query_community_level_available(check, {"community_levels": [6]})
        )
        check.community_level = 6
        self.assertTrue(
            mod._query_community_level_available(check, {"community_levels": [6]})
        )

    def test_graphrag_json_drift_query_applies_bounded_budget(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        args = argparse.Namespace(
            python="graph-python",
            question="question",
            working_dir=Path("workspace"),
            method="drift",
            data=None,
            community_level=2,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            drift_primer_folds=2,
            drift_k_followups=3,
            drift_depth=1,
        )
        completed = SimpleNamespace(returncode=0, stdout="{}", stderr="")

        with patch.object(mod.subprocess, "run", return_value=completed) as run:
            self.assertIs(mod._graphrag_query_json_subprocess(args, {}), completed)

        command = run.call_args.args[0]
        request = json.loads(command[3])
        self.assertEqual(
            request["drift_budget"],
            {"primer_folds": 2, "k_followups": 3, "n_depth": 1},
        )
        self.assertIn("config.drift_search.primer_folds", command[2])
        self.assertIn("config.drift_search.drift_k_followups", command[2])
        self.assertIn("config.drift_search.n_depth", command[2])

    def test_graphrag_community_filter_is_scoped_to_report_workflow(self) -> None:
        _load_script("query_graphrag_index.py")
        import pandas as pd

        from gems_rag.graphrag_indexing import _filtered_community_workflow

        observed = []

        class Reader:
            async def communities(self):
                return pd.DataFrame({"community": [10, 20, 30], "level": [1, 2, 3]})

        original_reader = Reader.communities

        async def workflow(_config, _context):
            frame = await Reader().communities()
            observed.extend(frame["community"].tolist())
            return "complete"

        wrapped = _filtered_community_workflow(workflow, Reader, frozenset({2}))

        self.assertEqual(asyncio.run(wrapped(None, None)), "complete")
        self.assertEqual(observed, [20])
        self.assertIs(Reader.communities, original_reader)

    def test_graphrag_truncated_index_cache_is_removed_before_retry(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            cache_dir = working_dir / "cache" / "extract_profile"
            cache_dir.mkdir(parents=True)
            (working_dir / "settings.yaml").write_text(
                "extract_graph:\n  model_instance_name: extract_profile\n",
                encoding="utf-8",
            )
            truncated = cache_dir / "truncated_v4"
            truncated.write_text(
                json.dumps(
                    {
                        "result": {
                            "response": {
                                "choices": [{"finish_reason": "length"}]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            complete = cache_dir / "complete_v4"
            complete.write_text(
                json.dumps(
                    {
                        "result": {
                            "response": {"choices": [{"finish_reason": "stop"}]}
                        }
                    }
                ),
                encoding="utf-8",
            )

            detected = mod._truncated_index_cache_entries(working_dir)
            removed = mod._remove_truncated_index_cache_entries(working_dir)

            self.assertEqual(detected, ["cache/extract_profile/truncated_v4"])
            self.assertEqual(removed, detected)
            self.assertFalse(truncated.exists())
            self.assertTrue(complete.exists())

    def test_graphrag_invalid_community_report_cache_is_removed(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            cache_dir = working_dir / "cache" / "report_profile"
            cache_dir.mkdir(parents=True)
            (working_dir / "settings.yaml").write_text(
                "community_reports:\n  model_instance_name: report_profile\n",
                encoding="utf-8",
            )

            def payload(findings):
                return {
                    "result": {
                        "response": {
                            "content": json.dumps({"findings": findings}),
                            "choices": [{"finish_reason": "stop"}],
                        }
                    }
                }

            valid_finding = {"summary": "Rule", "explanation": "Grounded detail"}
            valid = cache_dir / "valid_v4"
            valid.write_text(
                json.dumps(payload([valid_finding, valid_finding])),
                encoding="utf-8",
            )
            one_finding = cache_dir / "one_finding_v4"
            one_finding.write_text(
                json.dumps(payload([valid_finding])),
                encoding="utf-8",
            )
            empty = cache_dir / "empty_findings_v4"
            empty.write_text(json.dumps(payload([])), encoding="utf-8")
            malformed = cache_dir / "malformed_v4"
            malformed.write_text("not json", encoding="utf-8")

            detected = mod._invalid_community_report_cache_entries(working_dir)
            removed = mod._remove_invalid_community_report_cache_entries(working_dir)
            valid_exists = valid.exists()
            one_finding_exists = one_finding.exists()
            empty_exists = empty.exists()
            malformed_exists = malformed.exists()

        self.assertEqual(
            detected,
            [
                "cache/report_profile/empty_findings_v4",
                "cache/report_profile/malformed_v4",
            ],
        )
        self.assertEqual(removed, detected)
        self.assertTrue(valid_exists)
        self.assertTrue(one_finding_exists)
        self.assertFalse(empty_exists)
        self.assertFalse(malformed_exists)

    def test_graphrag_index_identity_tracks_indexing_prompts(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            (working_dir / "input").mkdir()
            (working_dir / "prompts").mkdir()
            (working_dir / "input" / "mutcd_chunks.txt").write_text("manual", encoding="utf-8")
            (working_dir / "settings.yaml").write_text("models: {}", encoding="utf-8")
            for name in mod.INDEX_PROMPT_NAMES:
                (working_dir / "prompts" / name).write_text(f"{name} v1", encoding="utf-8")
            args = argparse.Namespace(working_dir=working_dir, limit=None)

            before = mod._index_identity(args)
            (working_dir / "prompts" / "community_report_text.txt").write_text(
                "community_report_text.txt v2",
                encoding="utf-8",
            )
            after = mod._index_identity(args)

        self.assertNotEqual(before, after)
        self.assertEqual(set(before["index_prompts"]), set(mod.INDEX_PROMPT_NAMES))

    def test_paperqa_maps_selected_backend_key_to_openai_client(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        args = argparse.Namespace(
            api_key_env="LOCAL_OPENAI_API_KEY",
            allow_missing_api_key=True,
            base_url="http://localhost:8000/v1",
        )

        with patch.dict(os.environ, {}, clear=True):
            mod._ensure_api_key(args)
            self.assertEqual(os.environ["LOCAL_OPENAI_API_KEY"], "local")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "local")
            self.assertEqual(os.environ["OPENAI_BASE_URL"], "http://localhost:8000/v1")

    def test_paperqa_routes_custom_models_through_openai_compatible_provider(self) -> None:
        mod = _load_script("query_paperqa_index.py")

        self.assertEqual(
            mod._litellm_model("nomic-embed-text", "http://localhost:8000/v1"),
            "openai/nomic-embed-text",
        )
        self.assertEqual(
            mod._litellm_model("Qwen/Qwen3-8B", "http://localhost:8000/v1"),
            "openai/Qwen/Qwen3-8B",
        )
        self.assertEqual(
            mod._litellm_model("openai/custom", "http://localhost:8000/v1"),
            "openai/custom",
        )
        self.assertEqual(mod._litellm_model("gpt-4o-mini", None), "gpt-4o-mini")

    def test_lightrag_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            working_dir = root / "index"
            working_dir.mkdir()
            corpus = root / "corpus.txt"
            corpus.write_text("manual corpus", encoding="utf-8")
            args = argparse.Namespace(
                repo=repo,
                working_dir=working_dir,
                corpus=corpus,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                (working_dir / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

                mod.publish_completion_marker(
                    working_dir / mod.INDEX_SENTINEL,
                    mod._index_identity(args),
                    index_files=mod._index_files(working_dir),
                )
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

                corpus.write_text("changed corpus", encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

    def test_raganything_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_raganything_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "raganything"
            lightrag_repo = root / "lightrag"
            repo.mkdir()
            lightrag_repo.mkdir()
            working_dir = root / "index"
            working_dir.mkdir()
            content_list = root / "content.json"
            content_list.write_text("[]", encoding="utf-8")
            args = argparse.Namespace(
                repo=repo,
                lightrag_repo=lightrag_repo,
                working_dir=working_dir,
                content_list=content_list,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                (working_dir / "graph_chunk_entity_relation.graphml").write_text("<graphml />", encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

                mod.publish_completion_marker(
                    working_dir / mod.INDEX_SENTINEL,
                    mod._index_identity(args),
                    index_files=mod._index_files(working_dir),
                )
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

                args.limit = 1
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

                mod.publish_completion_marker(
                    working_dir / mod.INDEX_SENTINEL,
                    mod._index_identity(args),
                    index_files=mod._index_files(working_dir),
                )
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])

                content_list.write_text('[{"changed":true}]', encoding="utf-8")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

    def test_raganything_query_fails_closed_before_initialization(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(
            command="query",
            repo=Path("/tmp/raganything"),
            lightrag_repo=Path("/tmp/lightrag"),
        )
        report = {"runnable": False, "index_ready": False}
        with (
            patch.object(mod, "_add_repo"),
            patch.object(mod, "_dependency_report", return_value=report),
            patch.object(mod, "_make_rag") as make_rag,
            patch("builtins.print"),
        ):
            self.assertEqual(asyncio.run(mod._main(args)), 2)
        make_rag.assert_not_called()

    def test_raganything_context_query_disables_vlm_and_emits_contexts(self) -> None:
        mod = _load_script("query_raganything_index.py")
        args = argparse.Namespace(
            mode="hybrid",
            question="What does Section 2A.04 require?",
            top_k=5,
            chunk_top_k=7,
            only_need_context=True,
            response_type="Bullet Points",
        )

        self.assertEqual(
            mod._query_kwargs(args),
            {
                "top_k": 5,
                "chunk_top_k": 7,
                "only_need_context": True,
                "response_type": "Bullet Points",
                "vlm_enhanced": False,
            },
        )

        payload = mod._query_payload(args, "retrieved context")
        self.assertEqual(payload["result"], "retrieved context")
        self.assertEqual(payload["contexts"][0]["text"], "retrieved context")
        self.assertEqual(payload["contexts"][0]["metadata"]["top_k"], 5)
        self.assertEqual(payload["contexts"][0]["metadata"]["chunk_top_k"], 7)

    def test_raganything_query_ready_initializes_lightrag(self) -> None:
        mod = _load_script("query_raganything_index.py")

        class FakeRag:
            called = False

            async def _ensure_lightrag_initialized(self):
                self.called = True
                return {"success": True}

        rag = FakeRag()
        asyncio.run(mod._ensure_query_ready(rag))
        self.assertTrue(rag.called)

        class BrokenRag:
            async def _ensure_lightrag_initialized(self):
                return {"success": False, "error": "missing index"}

        with self.assertRaisesRegex(RuntimeError, "missing index"):
            asyncio.run(mod._ensure_query_ready(BrokenRag()))

    def test_raganything_skips_parser_only_when_parsing_is_not_requested(self) -> None:
        mod = _load_script("query_raganything_index.py")

        shared = SimpleNamespace(_parser_installation_checked=False)
        mod._skip_parser_check_for_preparsed_input(
            shared,
            argparse.Namespace(command="index", ingestion_mode="shared_corpus"),
        )
        self.assertTrue(shared._parser_installation_checked)

        query = SimpleNamespace(_parser_installation_checked=False)
        mod._skip_parser_check_for_preparsed_input(
            query,
            argparse.Namespace(command="query", ingestion_mode="native_pdf"),
        )
        self.assertTrue(query._parser_installation_checked)

        native_index = SimpleNamespace(_parser_installation_checked=False)
        mod._skip_parser_check_for_preparsed_input(
            native_index,
            argparse.Namespace(command="index", ingestion_mode="native_pdf"),
        )
        self.assertFalse(native_index._parser_installation_checked)

    def test_graphrag_index_files_ignore_prepared_input(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            working_dir = Path(td)
            (working_dir / "settings.yaml").write_text("settings", encoding="utf-8")
            (working_dir / "input").mkdir()
            (working_dir / "input" / "mutcd_chunks.txt").write_text("prepared", encoding="utf-8")
            self.assertEqual(mod._index_files(working_dir), [])

            output = working_dir / "output" / "artifacts"
            output.mkdir(parents=True)
            (output / "create_final_nodes.parquet").write_text("indexed", encoding="utf-8")
            index_files = mod._index_files(working_dir)
            self.assertEqual(index_files, ["output/artifacts/create_final_nodes.parquet"])

            args = argparse.Namespace(working_dir=working_dir)
            sentinel = working_dir / mod.INDEX_SENTINEL
            self.assertFalse(mod.completion_marker_matches(sentinel, mod._index_identity(args)))
            mod.publish_completion_marker(
                sentinel,
                mod._index_identity(args),
                index_files=index_files,
            )
            marker = mod.read_completion_marker(sentinel)
            self.assertTrue(mod.completion_marker_matches(sentinel, mod._index_identity(args)))
            self.assertTrue(mod._sentinel_files_present(marker, index_files))
            self.assertTrue(mod._index_ready(args))

            smoke_args = argparse.Namespace(working_dir=working_dir, limit=1)
            self.assertFalse(mod._index_ready(smoke_args))

            (working_dir / "input" / "mutcd_chunks.txt").write_text("changed", encoding="utf-8")
            self.assertFalse(mod.completion_marker_matches(sentinel, mod._index_identity(args)))

    def test_graphrag_prepare_is_atomic_and_limited(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                "\n".join(
                    [
                        json.dumps({"doc_id": "one", "text": "first"}),
                        json.dumps({"doc_id": "two", "text": "second"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                working_dir=root / "index",
                chunks=chunks,
                force=False,
                limit=1,
            )

            with patch("builtins.print"):
                self.assertEqual(mod._prepare(args), 0)

            prepared = (args.working_dir / "input" / "mutcd_chunks.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("--- one ---", prepared)
            self.assertNotIn("--- two ---", prepared)
            self.assertFalse(
                (args.working_dir / "input" / ".mutcd_chunks.txt.tmp").exists()
            )

            full_args = argparse.Namespace(working_dir=args.working_dir, limit=None)
            smoke_args = argparse.Namespace(working_dir=args.working_dir, limit=1)
            self.assertNotEqual(
                mod._index_identity(full_args),
                mod._index_identity(smoke_args),
            )

    def test_interrupted_api_backed_indexes_clear_completion_markers(self) -> None:
        lightrag = _load_script("query_lightrag_index.py")
        raganything = _load_script("query_raganything_index.py")
        graphrag = _load_script("query_graphrag_index.py")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            light_dir = root / "light"
            light_dir.mkdir()
            light_marker = light_dir / lightrag.INDEX_SENTINEL
            light_marker.write_text('{"complete":true}\n', encoding="utf-8")
            corpus = root / "corpus.txt"
            corpus.write_text("manual", encoding="utf-8")

            class BrokenLightRag:
                async def initialize_storages(self):
                    return None

                async def ainsert(self, _corpus):
                    raise RuntimeError("interrupted")

                async def finalize_storages(self):
                    return None

            light_args = argparse.Namespace(
                command="index",
                repo=root,
                working_dir=light_dir,
                corpus=corpus,
            )
            with (
                patch.object(lightrag, "_add_repo"),
                patch.object(lightrag, "_api_key", return_value="local"),
                patch.object(lightrag, "_make_rag", return_value=BrokenLightRag()),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                asyncio.run(lightrag._main(light_args))
            self.assertFalse(light_marker.exists())

            rag_dir = root / "raganything"
            rag_dir.mkdir()
            rag_marker = rag_dir / raganything.INDEX_SENTINEL
            rag_marker.write_text('{"complete":true}\n', encoding="utf-8")
            content_list = root / "content.json"
            content_list.write_text("[]", encoding="utf-8")

            class BrokenRagAnything:
                async def insert_content_list(self, **_kwargs):
                    raise RuntimeError("interrupted")

            rag_args = argparse.Namespace(
                command="index",
                repo=root,
                lightrag_repo=root,
                working_dir=rag_dir,
                ingestion_mode="shared_corpus",
                content_list=content_list,
                file_path="manual.pdf",
                doc_id="manual",
                display_stats=False,
            )
            with (
                patch.object(raganything, "_add_repo"),
                patch.object(raganything, "_api_key", return_value="local"),
                patch.object(raganything, "_make_rag", return_value=BrokenRagAnything()),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                asyncio.run(raganything._main(rag_args))
            self.assertFalse(rag_marker.exists())

            graph_dir = root / "graph"
            graph_dir.mkdir()
            graph_marker = graph_dir / graphrag.INDEX_SENTINEL
            graph_marker.write_text('{"complete":true}\n', encoding="utf-8")
            graph_args = argparse.Namespace(working_dir=graph_dir, method="standard")
            with patch.object(graphrag, "_run_graphrag", return_value=2):
                self.assertEqual(graphrag._index(graph_args, {}), 2)
            self.assertFalse(graph_marker.exists())

    def test_lightrag_skips_insert_when_canonical_document_is_processed(self) -> None:
        mod = _load_script("query_lightrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            working_dir = root / "index"
            working_dir.mkdir()
            (working_dir / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")
            corpus = root / "corpus.txt"
            corpus.write_text("complete corpus", encoding="utf-8")

            class ProcessedStatus:
                async def get_by_id(self, _doc_id):
                    return {"status": "processed"}

            class CompleteLightRag:
                doc_status = ProcessedStatus()

                async def initialize_storages(self):
                    return None

                async def ainsert(self, _corpus):
                    raise AssertionError("processed corpus must not be inserted again")

                async def finalize_storages(self):
                    return None

            args = argparse.Namespace(
                command="index",
                repo=root,
                working_dir=working_dir,
                corpus=corpus,
            )
            with (
                patch.object(mod, "_add_repo"),
                patch.object(mod, "_api_key", return_value="local"),
                patch.object(mod, "_make_rag", return_value=CompleteLightRag()),
                patch("builtins.print"),
            ):
                self.assertEqual(asyncio.run(mod._main(args)), 0)

            self.assertTrue((working_dir / mod.INDEX_SENTINEL).exists())

    def test_silent_lightrag_failures_do_not_publish_completion_markers(self) -> None:
        lightrag = _load_script("query_lightrag_index.py")
        raganything = _load_script("query_raganything_index.py")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corpus = root / "corpus.txt"
            corpus.write_text("manual", encoding="utf-8")
            light_dir = root / "light"
            light_dir.mkdir()

            class FailedCounts:
                async def get_status_counts(self):
                    return {"processed": 0, "failed": 1}

            class SilentFailedLightRag:
                doc_status = FailedCounts()

                async def initialize_storages(self):
                    return None

                async def ainsert(self, _corpus):
                    return "track-id"

                async def finalize_storages(self):
                    return None

            light_args = argparse.Namespace(
                command="index",
                repo=root,
                working_dir=light_dir,
                corpus=corpus,
            )
            with (
                patch.object(lightrag, "_add_repo"),
                patch.object(lightrag, "_api_key", return_value="local"),
                patch.object(
                    lightrag, "_make_rag", return_value=SilentFailedLightRag()
                ),
            ):
                self.assertEqual(asyncio.run(lightrag._main(light_args)), 2)
            self.assertFalse((light_dir / lightrag.INDEX_SENTINEL).exists())

            rag_dir = root / "raganything"
            rag_dir.mkdir()
            content_list = root / "content.json"
            content_list.write_text("[]", encoding="utf-8")

            class FailedDocumentStatus:
                async def get_by_id(self, _doc_id):
                    return {"status": "failed"}

            class EmbeddedLightRag:
                doc_status = FailedDocumentStatus()

                async def ainsert(self, **_kwargs):
                    return "track-id"

            class SilentFailedRagAnything:
                lightrag = EmbeddedLightRag()

                async def _ensure_lightrag_initialized(self):
                    return {"success": True}

                async def insert_content_list(self, **kwargs):
                    await self.lightrag.ainsert(input="manual", ids=kwargs["doc_id"])

            rag_args = argparse.Namespace(
                command="index",
                repo=root,
                lightrag_repo=root,
                working_dir=rag_dir,
                ingestion_mode="shared_corpus",
                content_list=content_list,
                file_path="manual.pdf",
                doc_id="manual",
                display_stats=False,
            )
            with (
                patch.object(raganything, "_add_repo"),
                patch.object(raganything, "_api_key", return_value="local"),
                patch.object(
                    raganything,
                    "_make_rag",
                    return_value=SilentFailedRagAnything(),
                ),
                self.assertRaisesRegex(RuntimeError, "processing incomplete"),
            ):
                asyncio.run(raganything._main(rag_args))
            self.assertFalse((rag_dir / raganything.INDEX_SENTINEL).exists())

    def test_graphrag_json_contexts_are_harness_native_and_capped(self) -> None:
        mod = _load_script("query_graphrag_index.py")
        args = argparse.Namespace(
            question="What does Section 2A.04 require?",
            method="local",
            top_k=2,
            response_type="Multiple Paragraphs",
            community_level=2,
            dynamic_community_selection=False,
        )
        stdout = json.dumps(
            {
                "response": "GraphRAG answer",
                "context_data": {
                    "sources": [
                        {
                            "id": "source-1",
                            "text": "source text",
                            "section_id": "2A.04",
                            "rank": 3,
                        }
                    ],
                    "reports": [
                        {
                            "title": "Warning Signs",
                            "content": "report content",
                            "rank": 2,
                        }
                    ],
                    "relationships": [
                        {
                            "source": "A",
                            "target": "B",
                            "description": "relationship text",
                        }
                    ],
                },
            }
        )

        payload = mod._query_payload_from_stdout(args, stdout)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["result"], "GraphRAG answer")
        self.assertEqual(len(payload["contexts"]), 2)
        self.assertEqual(payload["contexts"][0]["name"], "source-1")
        self.assertEqual(payload["contexts"][0]["kind"], "chunk")
        self.assertEqual(payload["contexts"][0]["text"], "source text")
        self.assertEqual(payload["contexts"][0]["score"], 3.0)
        self.assertEqual(payload["contexts"][0]["metadata"]["graph_group"], "sources")
        self.assertIn("Warning Signs", payload["contexts"][1]["text"])
        self.assertEqual(payload["contexts"][1]["metadata"]["graph_group"], "reports")
        self.assertEqual(mod._contexts_from_graphrag_data(json.loads(stdout)["context_data"], top_k=0, method="local"), [])

    def test_paperqa_check_requires_index_for_runnable(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            repo.mkdir()
            index = root / "docs.pkl"
            chunks = root / "chunks.jsonl"
            chunks.write_text('{"text":"manual chunk"}\n', encoding="utf-8")
            args = argparse.Namespace(
                repo=repo,
                index=index,
                chunks=chunks,
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
            )
            with patch.object(mod, "_import_errors", return_value={}), patch.dict(os.environ, {}, clear=True):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertFalse(report["index_ready"])
                self.assertFalse(report["runnable"])

                index.write_bytes(b"pickle")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

                mod.publish_completion_marker(
                    mod._index_sentinel(index),
                    mod._index_identity(args),
                    index=mod.file_identity(index),
                )
                report = mod._dependency_report(args)
                self.assertTrue(report["index_ready"])
                self.assertTrue(report["runnable"])

                index.write_bytes(b"corrupted")
                report = mod._dependency_report(args)
                self.assertFalse(report["index_ready"])

    def test_paperqa_query_budget_and_context_records_are_harness_native(self) -> None:
        mod = _load_script("query_paperqa_index.py")
        settings = SimpleNamespace(answer=SimpleNamespace(evidence_k=10, answer_max_sources=5))
        mod._apply_query_budget(settings, argparse.Namespace(top_k=7))
        self.assertEqual(settings.answer.evidence_k, 7)
        self.assertEqual(settings.answer.answer_max_sources, 7)

        source = SimpleNamespace(
            text="original source chunk",
            name="MUTCD11e_2A04_Standard_13",
            doc=SimpleNamespace(docname="MUTCD 11th Edition", dockey="mutcd11e", citation="MUTCD"),
            model_extra={"section_id": "2A.04", "content_type": "Standard", "ordinal": 13, "page_printed": "31"},
        )
        context = SimpleNamespace(id="pqac-test", context="relevant summary", text=source, score=8)
        record = mod._paperqa_context_to_record(context)

        self.assertEqual(record["name"], "pqac-test")
        self.assertEqual(record["kind"], "chunk")
        self.assertIn("relevant summary", record["text"])
        self.assertIn("original source chunk", record["text"])
        self.assertEqual(record["score"], 8)
        self.assertEqual(record["metadata"]["source_name"], "MUTCD11e_2A04_Standard_13")
        self.assertEqual(record["metadata"]["section_id"], "2A.04")
        self.assertEqual(record["metadata"]["docname"], "MUTCD 11th Edition")

    def test_paperqa_pickle_replacement_is_atomic_on_failure(self) -> None:
        mod = _load_script("query_paperqa_index.py")

        class Unpickleable:
            def __reduce__(self):
                raise RuntimeError("cannot pickle")

        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "docs.pkl"
            index.write_bytes(b"previous-complete-index")
            with self.assertRaisesRegex(RuntimeError, "cannot pickle"):
                mod._write_pickle_atomic(index, Unpickleable())

            self.assertEqual(index.read_bytes(), b"previous-complete-index")
            self.assertEqual([path for path in index.parent.iterdir() if path != index], [])

    def test_hipporag_sidecar_enriches_contexts_without_counting_as_index(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_dir = root / "hipporag"
            chunks = root / "chunks.jsonl"
            row = {
                "doc_id": "MUTCD11e_2A04_Standard_13",
                "title": "Section 2A.04 Standard 13 - General",
                "text": "chunk text",
                "metadata": {"section_id": "2A.04", "page_printed": "31", "content_type": "Standard"},
            }
            chunks.write_text(json.dumps(row) + "\n", encoding="utf-8")

            sidecar = mod._write_metadata_sidecar(save_dir, [row])
            self.assertTrue(sidecar.exists())
            self.assertEqual(mod._index_files(save_dir), [])

            (save_dir / "hipporag_index.bin").write_text("indexed", encoding="utf-8")
            self.assertEqual(mod._index_files(save_dir), ["hipporag_index.bin"])

            manifest = mod._load_metadata_by_text(save_dir, chunks)
            context = mod._context_from_hit("chunk text", 0.75, 1, manifest)
            self.assertEqual(context["name"], "MUTCD11e_2A04_Standard_13")
            self.assertEqual(context["kind"], "chunk")
            self.assertEqual(context["score"], 0.75)
            self.assertEqual(context["metadata"]["section_id"], "2A.04")
            self.assertEqual(context["metadata"]["title"], "Section 2A.04 Standard 13 - General")

    def test_megarag_rejects_silent_lightrag_failure(self) -> None:
        mod = _load_script("query_megarag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            working_dir = root / "index"
            working_dir.mkdir()
            sentinel = working_dir / mod.INDEX_SENTINEL
            sentinel.write_text('{"complete":true}\n', encoding="utf-8")
            pages_content = root / "pages.json"
            pages_content.write_text("{}", encoding="utf-8")

            class FailedStatus:
                async def get_status_counts(self):
                    return {"processed": 0, "failed": 1}

            class SilentFailedMegaRag:
                doc_status = FailedStatus()

                async def ainsert(self, **_kwargs):
                    return None

                async def finalize_storages(self):
                    return None

            args = argparse.Namespace(
                working_dir=working_dir,
                pages_content=pages_content,
                mrag_dir=root,
                force=False,
                api_key_env="LOCAL_OPENAI_API_KEY",
                allow_missing_api_key=True,
            )
            report = {
                "environment_ready": True,
                "api_key_usable": True,
                "input_ready": True,
                "index_ready": False,
            }
            with (
                patch.object(mod, "_dependency_report", return_value=report),
                patch.object(
                    mod,
                    "_initialize_rag",
                    return_value=(SilentFailedMegaRag(), "tokens"),
                ),
            ):
                self.assertEqual(asyncio.run(mod._index(args)), 2)

            self.assertFalse(sentinel.exists())

    def test_hipporag_requires_matching_completion_sentinel(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"
            (repo / "src" / "hipporag").mkdir(parents=True)
            chunks = root / "chunks.jsonl"
            chunks.write_text('{"text":"chunk"}\n', encoding="utf-8")
            save_dir = root / "index"
            save_dir.mkdir()
            (save_dir / "partial-cache.sqlite").write_text("partial", encoding="utf-8")
            args = argparse.Namespace(
                repo=repo,
                chunks=chunks,
                save_dir=save_dir,
                python=Path("missing-python"),
                api_key_env="OPENAI_API_KEY",
                allow_missing_api_key=True,
                base_url=None,
                llm_base_url=None,
                embedding_base_url=None,
                llm_model="gpt-4o-mini",
                embedding_model="text-embedding-3-small",
            )
            with patch.object(mod, "_import_errors", return_value={}):
                report = mod._dependency_report(args)
            self.assertFalse(report["index_ready"])

            sentinel = {**mod._index_identity(args, chunks), "complete": True, "indexed_docs": 1}
            mod._write_json_atomic(save_dir / mod.INDEX_SENTINEL, sentinel)
            with patch.object(mod, "_import_errors", return_value={}):
                report = mod._dependency_report(args)
            self.assertTrue(report["index_ready"])

            args.limit = 1
            limited_sentinel = {
                **mod._index_identity(args, chunks),
                "complete": True,
                "indexed_docs": 1,
            }
            mod._write_json_atomic(save_dir / mod.INDEX_SENTINEL, limited_sentinel)
            args.limit = None
            with patch.object(mod, "_import_errors", return_value={}):
                report = mod._dependency_report(args)
            self.assertFalse(report["index_ready"])

            args.limit = 1
            with patch.object(mod, "_import_errors", return_value={}):
                report = mod._dependency_report(args)
            self.assertTrue(report["index_ready"])

            chunks.write_text('{"text":"changed"}\n', encoding="utf-8")
            with patch.object(mod, "_import_errors", return_value={}):
                report = mod._dependency_report(args)
            self.assertFalse(report["index_ready"])

    def test_hipporag_maps_selected_api_key_for_upstream_openai_clients(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        args = argparse.Namespace(api_key_env="HIPPORAG_TEST_KEY", allow_missing_api_key=False)
        with patch.dict(os.environ, {"HIPPORAG_TEST_KEY": "secret"}, clear=True):
            self.assertEqual(mod._ensure_api_key(args), "secret")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "secret")

    def test_hipporag_applies_reasoning_effort_and_isolates_cache(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        updates = []
        llm_model = SimpleNamespace(
            llm_config=SimpleNamespace(generate_params={"model": "qwen3:0.6b"}),
            cache_file_name="/tmp/qwen3_cache.sqlite",
            batch_upsert_llm_config=updates.append,
        )
        rag = SimpleNamespace(llm_model=llm_model)
        fake_module = ModuleType("hipporag")
        fake_module.HippoRAG = lambda **_kwargs: rag
        args = SimpleNamespace(
            repo=Path("/tmp/hipporag"),
            save_dir=Path("/tmp/index"),
            api_key_env="LOCAL_OPENAI_API_KEY",
            allow_missing_api_key=False,
            base_url="http://localhost:11434/v1",
            llm_base_url=None,
            embedding_base_url=None,
            llm_model="qwen3:0.6b",
            embedding_model="nomic-embed-text",
            reasoning_effort="none",
        )

        with patch.dict(os.environ, {"LOCAL_OPENAI_API_KEY": "local-key"}, clear=True), patch.dict(
            sys.modules, {"hipporag": fake_module}
        ):
            result = mod._hipporag(args)

        self.assertIs(result, rag)
        self.assertEqual(
            updates,
            [{"generate_params": {"model": "qwen3:0.6b", "reasoning_effort": "none"}}],
        )
        self.assertEqual(llm_model.cache_file_name, "/tmp/qwen3_cache_reasoning_none.sqlite")

    def test_hipporag_rebuild_clears_ready_sentinel_before_upstream_failure(self) -> None:
        mod = _load_script("query_hipporag_index.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chunks = root / "chunks.jsonl"
            chunks.write_text('{"text":"chunk"}\n', encoding="utf-8")
            save_dir = root / "index"
            save_dir.mkdir()
            sentinel = save_dir / mod.INDEX_SENTINEL
            sentinel.write_text('{"complete":true}\n', encoding="utf-8")
            args = argparse.Namespace(chunks=chunks, save_dir=save_dir, limit=None)

            def interrupted_index(**kwargs):
                raise RuntimeError("interrupted")

            rag = SimpleNamespace(index=interrupted_index)
            report = {
                "environment_ready": True,
                "input_ready": True,
                "model_service_ready": True,
                "credential_available": True,
            }
            with (
                patch.object(mod, "_dependency_report", return_value=report),
                patch.object(mod, "_hipporag", return_value=rag),
            ):
                self.assertEqual(mod._index(args), 2)

            self.assertFalse(sentinel.exists())

    def test_hipporag_patch_atomically_publishes_persistent_artifacts(self) -> None:
        patch_text = (ROOT / "patches" / "hipporag-lazy-optional-backends.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('temp_path = f"{self.openie_results_path}.tmp"', patch_text)
        self.assertIn("os.replace(temp_path, self.openie_results_path)", patch_text)
        self.assertIn('temp_filename = f"{self._graph_pickle_filename}.tmp"', patch_text)
        self.assertIn("os.replace(temp_filename, self._graph_pickle_filename)", patch_text)
        self.assertIn('temp_filename = f"{self.filename}.tmp"', patch_text)
        self.assertIn("os.replace(temp_filename, self.filename)", patch_text)
        self.assertIn('+        "pyarrow>=17,<24",', patch_text)
        self.assertIn("+    return OpenAIEmbeddingModel", patch_text)
        self.assertNotIn(
            '+    assert False, f"Unknown embedding model name: {embedding_model_name}"',
            patch_text,
        )

    def test_vector_db_check_is_runnable_before_lazy_index_exists(self) -> None:
        mod = _load_script("query_vector_db.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mrag_dir = root / "MRAG"
            cache = mrag_dir / "mmrag_cache_v3"
            cache.mkdir(parents=True)
            chunks = cache / "chunks.jsonl"
            chunks.write_text("", encoding="utf-8")
            args = argparse.Namespace(mrag_dir=mrag_dir, path=root / "qdrant_hash_vector")

            with patch.object(mod.importlib.util, "find_spec", return_value=object()):
                report = mod._dependency_report(args)
                self.assertTrue(report["environment_ready"])
                self.assertTrue(report["runnable"])
                self.assertFalse(report["index_ready"])

                chunks.unlink()
                report = mod._dependency_report(args)
                self.assertFalse(report["environment_ready"])
                self.assertFalse(report["runnable"])

    def test_heavy_adapters_report_isolated_python_paths(self) -> None:
        for script, env_name in [
            ("query_mrag_reference.py", "mrag-reference"),
            ("query_hipporag_index.py", "hipporag"),
            ("query_visrag_index.py", "visrag"),
        ]:
            mod = _load_script(script)
            args = argparse.Namespace(
                python=ROOT / "data" / "working" / "venvs" / env_name / "bin" / "python",
                repo=ROOT / "external",
                save_dir=ROOT / "data" / "working" / "hipporag_index",
                mrag_dir=ROOT / "data",
                manifest=ROOT / "missing-manifest.jsonl",
                embeddings=ROOT / "missing-embeddings.npy",
                model_name_or_path="local-model",
            )
            report = mod._dependency_report(args)
            self.assertEqual(report["adapter_python"], str(args.python))
            self.assertIn("adapter_python_found", report)
            self.assertIn("current_python", report)

    def test_heavy_adapter_reexec_is_noop_when_python_missing(self) -> None:
        mod = _load_script("query_visrag_index.py")
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(mod._maybe_reexec(Path(td) / "missing-python"))


if __name__ == "__main__":
    unittest.main()
