import io
import json
import unittest
from unittest.mock import patch

from loom_attention.two_gpu_smoke import (
    BenchmarkConfig,
    _flashinfer_attention_state,
    _merge_states,
    main,
    percentile,
    projected_transfer_bytes,
)


class TwoGpuSmokeContractTest(unittest.TestCase):
    def test_projected_bytes_show_route_query_asymmetry(self) -> None:
        config = BenchmarkConfig(
            prefix_tokens=4096,
            rows=1,
            query_heads=32,
            kv_heads=8,
            head_dim=128,
            dtype="float16",
        )
        payload = projected_transfer_bytes(config)
        self.assertEqual(payload["query"], 8192)
        self.assertEqual(payload["output"], 8192)
        self.assertEqual(payload["logsumexp"], 128)
        self.assertEqual(payload["attention_state"], 8320)
        self.assertEqual(payload["route_query_total"], 16512)
        self.assertEqual(payload["stage_kv_total"], 16_777_216)
        self.assertLess(payload["route_query_total"], payload["stage_kv_total"])

    def test_rejects_invalid_gqa_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "kv_heads must divide query_heads"):
            BenchmarkConfig(query_heads=12, kv_heads=5).validate()

    def test_rejects_unknown_attention_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported attention backend"):
            BenchmarkConfig(attention_backend="unknown").validate()

    def test_default_tolerance_tracks_wire_dtype(self) -> None:
        fp16 = BenchmarkConfig(dtype="float16")
        bf16 = BenchmarkConfig(dtype="bfloat16")
        self.assertEqual((fp16.atol, fp16.rtol), (2e-3, 2e-3))
        self.assertEqual((bf16.atol, bf16.rtol), (2e-2, 2e-2))

    def test_percentile_interpolates_ordered_samples(self) -> None:
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.99), 2.98)

    def test_plan_command_does_not_import_torch(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output):
            status = main(
                [
                    "plan",
                    "--prefix-tokens",
                    "128",
                    "--query-heads",
                    "8",
                    "--kv-heads",
                    "2",
                    "--head-dim",
                    "64",
                    "--attention-backend",
                    "flashinfer",
                ]
            )
        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["workload"]["prefix_tokens"], 128)
        self.assertEqual(report["workload"]["attention_backend"], "flashinfer")
        self.assertGreater(report["payload_bytes"]["stage_kv_total"], 0)

    def test_flashinfer_adapter_uses_native_state_and_merge_contract(self) -> None:
        class FakeTensor:
            def __init__(self, name: str) -> None:
                self.name = name

            def contiguous(self):
                return self

            def float(self):
                return self

        class FakeTorch:
            def __init__(self) -> None:
                self.stack_dimensions = []

            def stack(self, values, dim=0):
                self.stack_dimensions.append((len(values), dim))
                return FakeTensor("stack")

        class FakeFlashInfer:
            def __init__(self) -> None:
                self.decode_calls = []
                self.merge_calls = []

            def single_decode_with_kv_cache(self, query, key, value, **kwargs):
                self.decode_calls.append((query, key, value, kwargs))
                return FakeTensor("output"), FakeTensor("lse")

            def merge_states(self, output, logsumexp):
                self.merge_calls.append((output, logsumexp))
                return FakeTensor("merged-output"), FakeTensor("merged-lse")

        torch = FakeTorch()
        flashinfer = FakeFlashInfer()
        query = [FakeTensor("q0"), FakeTensor("q1")]
        key = FakeTensor("key")
        value = FakeTensor("value")
        with patch(
            "loom_attention.two_gpu_smoke._load_flashinfer",
            return_value=flashinfer,
        ):
            state = _flashinfer_attention_state(
                torch, query, key, value, scale=0.125
            )
            merged = _merge_states(torch, [state, state], backend="flashinfer")

        self.assertEqual(len(flashinfer.decode_calls), 2)
        self.assertEqual(
            flashinfer.decode_calls[0][3],
            {
                "kv_layout": "NHD",
                "pos_encoding_mode": "NONE",
                "sm_scale": 0.125,
                "return_lse": True,
            },
        )
        self.assertEqual(torch.stack_dimensions, [(2, 0), (2, 0), (2, 1), (2, 1)])
        self.assertEqual(len(flashinfer.merge_calls), 1)
        self.assertEqual(merged[0].name, "merged-output")
        self.assertEqual(merged[1].name, "merged-lse")

    def test_run_reports_environment_failure_as_exit_two(self) -> None:
        error = io.StringIO()
        with (
            patch(
                "loom_attention.two_gpu_smoke._run",
                side_effect=RuntimeError("CUDA unavailable"),
            ),
            patch("sys.stderr", error),
        ):
            status = main(["run", "--iterations", "1", "--warmup", "0"])
        self.assertEqual(status, 2)
        self.assertIn("CUDA unavailable", error.getvalue())


if __name__ == "__main__":
    unittest.main()
