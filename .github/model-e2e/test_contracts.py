import copy
import fcntl
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from contracts import (
    CASES,
    PACKAGES,
    REPOSITORIES,
    digest,
    suite_passed,
    validate_answer,
    validate_lora,
    validate_request,
    verify_candidate,
    write_json,
)
from host import docker_command, export_evidence
from recipes import training_config
from resource_queue import ResourceUnavailable, gpu_busy, reservation


def request():
    return {
        "schema": 1,
        "mode": "pr",
        "harness_sha": "a" * 40,
        "pr": {"repository": REPOSITORIES["sglang"], "number": 88, "sha": "b" * 40},
        "sources": {
            key: {"repository": repo, "sha": "b" * 40}
            for key, repo in REPOSITORIES.items()
        },
        "build_run_id": 123,
        "build_workflow_sha": "c" * 40,
    }


def rank(index=0):
    return {
        "rank": index,
        "global_step": 1,
        "optimizer_steps": 1,
        "train_end": True,
        "raw_losses": [2.5],
    }


def answer(text="Paris", reason="stop", tokens=4):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": reason}],
        "usage": {"completion_tokens": tokens},
    }


class Contracts(unittest.TestCase):
    def test_pypi_default(self):
        validate_request(
            {
                "schema": 1,
                "mode": "pypi",
                "version": "latest",
                "expected_version": "0.7.0.post3",
                "harness_sha": "a" * 40,
            }
        )

    def test_pypi_without_expected_version_rejected(self):
        with self.assertRaises(ValueError):
            validate_request(
                {
                    "schema": 1,
                    "mode": "pypi",
                    "version": "latest",
                    "harness_sha": "a" * 40,
                }
            )

    def test_pr_snapshot(self):
        validate_request(request())

    def test_changed_pr_head(self):
        data = request()
        data["pr"]["sha"] = "c" * 40
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_foreign_repository(self):
        data = request()
        data["pr"]["repository"] = "attacker/fork"
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_short_sha(self):
        data = request()
        data["harness_sha"] = "abc123"
        with self.assertRaises(ValueError):
            validate_request(data)

    def test_loss_only_requires_one_finite_step(self):
        self.assertEqual(validate_lora([rank()], 1)["status"], "passed")

    def test_all_ranks_required(self):
        with self.assertRaises(ValueError):
            validate_lora([rank()], 8)

    def test_duplicate_rank(self):
        with self.assertRaises(ValueError):
            validate_lora([rank(), rank()], 2)

    def test_bad_lora_evidence(self):
        changes = [
            {"global_step": 0},
            {"global_step": 2},
            {"optimizer_steps": 0},
            {"train_end": False},
            {"raw_losses": []},
            {"raw_losses": [float("nan")]},
            {"raw_losses": [float("inf")]},
            {"raw_losses": [True]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_lora([rank() | change], 1)

    def test_answer_semantic_smoke(self):
        self.assertEqual(validate_answer(answer())["status"], "passed")
        validate_answer(answer("巴黎"))

    def test_bad_answers(self):
        for value in [
            answer(""),
            answer(None),
            answer("London"),
            answer("Paris\ufffd"),
            answer("Paris\x00"),
            answer(reason="length"),
            answer(tokens=0),
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_answer(value)

    def test_incomplete_suite_never_green(self):
        complete = {case: {"status": "passed"} for case in CASES}
        self.assertTrue(suite_passed(complete))
        del complete[CASES[0]]
        self.assertFalse(suite_passed(complete))

    def test_recipes_have_approved_boundary(self):
        for case in CASES[:2]:
            config = training_config(case, "/tmp/test")
            self.assertEqual(config["max_steps"], 1)
            self.assertEqual(config["gradient_accumulation_steps"], 1)
            self.assertFalse(config["logging_nan_inf_filter"])
            self.assertEqual(config["finetuning_type"], "lora")
            self.assertTrue(config["use_kt"])
            self.assertEqual(config["kt_config"]["kt_num_gpu_experts"], 0)


class Candidates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = request()
        self.manifest = {
            "schema": 1,
            "sources": self.request["sources"],
            "build_run_id": 123,
            "build_workflow_sha": "c" * 40,
            "wheels": {},
        }
        for package in PACKAGES:
            name = package.replace("-", "_")
            path = self.root / f"{name}-1.0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as wheel:
                wheel.writestr(
                    f"{name}-1.0.dist-info/METADATA", f"Name: {package}\nVersion: 1.0\n"
                )
            self.manifest["wheels"][package] = {
                "file": path.name,
                "version": "1.0",
                "sha256": digest(path),
            }
        self.save()

    def save(self):
        write_json(self.root / "candidate.json", self.manifest)

    def test_final_five_wheels(self):
        verify_candidate(self.root, self.request)

    def test_stale_main_snapshot(self):
        changed = copy.deepcopy(self.request)
        changed["sources"]["accelerate"]["sha"] = "d" * 40
        with self.assertRaises(ValueError):
            verify_candidate(self.root, changed)

    def test_wrong_build(self):
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request | {"build_run_id": 999})

    def test_tampered_wheel(self):
        path = self.root / self.manifest["wheels"]["ktransformers"]["file"]
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request)

    def test_path_traversal(self):
        self.manifest["wheels"]["ktransformers"]["file"] = "../escape.whl"
        self.save()
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request)

    def test_symlink_wheel(self):
        path = self.root / self.manifest["wheels"]["ktransformers"]["file"]
        renamed = path.with_suffix(".real")
        path.rename(renamed)
        path.symlink_to(renamed)
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request)

    def test_wrong_metadata_version(self):
        self.manifest["wheels"]["ktransformers"]["version"] = "999"
        self.save()
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request)

    def test_raw_native_extra_rejected(self):
        (self.root / "sgl_kernel_kt-1.0.whl").write_bytes(b"raw")
        with self.assertRaises(ValueError):
            verify_candidate(self.root, self.request)


class QueueAndSandbox(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_busy_then_idle_waits(self):
        states = iter([(True, {}), (False, {}), (False, {})])
        with patch("resource_queue.time.sleep") as sleep:
            with reservation(
                self.root / "lock",
                self.root / "queue",
                interval=0,
                idle_samples=2,
                probe=lambda: next(states),
            ):
                self.assertEqual(sleep.call_count, 2)
        self.assertTrue((self.root / "lock").exists())

    def test_queue_timeout_is_not_a_model_failure(self):
        with self.assertRaises(ResourceUnavailable):
            with reservation(self.root / "lock", self.root / "queue", timeout=0):
                self.fail("should not start")

    def test_lock_held_through_work_and_released_on_exception(self):
        path = self.root / "lock"
        with self.assertRaisesRegex(RuntimeError, "owned failure"):
            with reservation(
                path, self.root / "queue", idle_samples=1, probe=lambda: (False, {})
            ):
                with path.open("a") as other:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise RuntimeError("owned failure")
        with path.open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_compute_process_always_queues(self):
        with patch(
            "resource_queue.subprocess.check_output", side_effect=["123\n", "2, 0\n"]
        ):
            self.assertTrue(gpu_busy()[0])

    def test_non_compute_vram_queues(self):
        with patch(
            "resource_queue.subprocess.check_output", side_effect=["", "4096, 0\n"]
        ):
            self.assertTrue(gpu_busy()[0])

    def test_idle(self):
        with patch(
            "resource_queue.subprocess.check_output", side_effect=["", "2, 0\n2, 0\n"]
        ):
            self.assertFalse(gpu_busy()[0])

    def test_driver_error_never_means_idle(self):
        with patch(
            "resource_queue.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                gpu_busy()

    def test_unknown_stats_never_mean_idle(self):
        with patch(
            "resource_queue.subprocess.check_output", side_effect=["", "N/A, N/A\n"]
        ):
            with self.assertRaises(ResourceUnavailable):
                gpu_busy()

    def test_container_has_no_host_secrets_or_privileges(self):
        config = {
            "image": "test@sha256:" + "a" * 64,
            "models": {"glm53": {"path": "/mnt/models/glm53"}},
        }
        command = docker_command(
            config, "/harness", "/input", Path("/output"), "owned-name"
        )
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--read-only", command)
        self.assertIn("--no-healthcheck", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--network=host", command)
        self.assertNotIn("--ipc=host", command)
        self.assertIn(
            "type=bind,src=/mnt/models/glm53,dst=/models/glm53,readonly", command
        )
        self.assertNotIn("docker.sock", " ".join(command))
        self.assertIn("--device=nvidia.com/gpu=all", command)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertNotIn("--gpus=all", command)

    def test_rootless_user_mapping(self):
        config = {"rootless": True, "image": "test@sha256:" + "a" * 64, "models": {}}
        command = docker_command(
            config, "/harness", "/input", Path("/output"), "owned-name"
        )
        self.assertEqual(command[command.index("--user") + 1], "0:0")

    def test_evidence_export_refuses_symlinks(self):
        source, destination = self.root / "source", self.root / "export"
        source.mkdir()
        destination.mkdir()
        (source / "leak").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            export_evidence(source, destination)

    def test_regular_evidence_export(self):
        source, destination = self.root / "source", self.root / "export"
        source.mkdir()
        destination.mkdir()
        write_json(source / "suite.json", {"status": "failed"})
        export_evidence(source, destination)
        self.assertEqual(
            json.loads((destination / "suite.json").read_text())["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
