import tempfile
import unittest
from pathlib import Path

from scripts.summarize import load_rows, summarize


# The PyTorch row deliberately carries a different device string than the HIP
# rows. On real hardware the HIP backend reports "AMD Radeon RX 9070 XT" and
# PyTorch reports "AMD Radeon Graphics" for the same GPU, so grouping must key on
# architecture for the SDPA comparison to join at all.
CSV = """device,architecture,backend,algorithm,dtype,batch,heads,sequence,dimension,causal,average_ms,median_ms,p95_ms,tflops,logical_bandwidth_gbs,workspace_bytes,max_abs_error,max_rel_error
AMD Radeon RX 9070 XT,gfx1201,hip,materialized,fp32,1,8,128,64,0,2.1,2.0,2.3,1.0,1.0,524288,0.0001,0.001
AMD Radeon RX 9070 XT,gfx1201,hip,fused,fp32,1,8,128,64,0,1.6,1.5,1.8,2.0,2.0,0,0.0002,0.002
AMD Radeon RX 9070 XT,gfx1201,hip,fused-cached,fp32,1,8,128,64,0,1.1,1.0,1.2,2.0,2.0,0,0.00015,0.0015
AMD Radeon Graphics,gfx1201,pytorch,sdpa-auto,fp32,1,8,128,64,0,0.9,0.8,1.0,2.0,2.0,-1,,
"""


class SummarizeTests(unittest.TestCase):
    def test_computes_speedup_and_workspace_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(CSV, encoding="utf-8")
            report = summarize(load_rows(path))

        self.assertIn("2.00x", report)
        self.assertIn("524,288 B", report)
        self.assertIn("0.0002", report)
        self.assertIn("1.2000 ms", report)
        self.assertIn("0.8000 ms", report)
        self.assertIn("0.80x", report)
        # The Device column shows the custom-kernel device, not the PyTorch one.
        self.assertIn("AMD Radeon RX 9070 XT", report)

    def test_rejects_unpaired_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "materialized, fused, and fused-cached"):
            summarize([
                {
                    "device": "GPU",
                    "architecture": "gfx1201",
                    "dtype": "fp32",
                    "batch": "1",
                    "heads": "1",
                    "sequence": "8",
                    "dimension": "8",
                    "causal": "0",
                    "algorithm": "fused",
                    "average_ms": "1",
                    "workspace_bytes": "0",
                    "max_abs_error": "0",
                }
            ])


if __name__ == "__main__":
    unittest.main()
