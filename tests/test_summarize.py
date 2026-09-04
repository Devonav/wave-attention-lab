import tempfile
import unittest
from pathlib import Path

from scripts.summarize import load_rows, summarize


CSV = """device,backend,algorithm,batch,heads,sequence,dimension,causal,average_ms,tflops,logical_bandwidth_gbs,workspace_bytes,max_abs_error,max_rel_error
RX 9070 XT,hip,materialized,1,8,128,64,0,2.0,1.0,1.0,524288,0.0001,0.001
RX 9070 XT,hip,fused,1,8,128,64,0,1.0,2.0,2.0,0,0.0002,0.002
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

    def test_rejects_unpaired_results(self) -> None:
        with self.assertRaisesRegex(ValueError, "both materialized and fused"):
            summarize([
                {
                    "device": "GPU",
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

