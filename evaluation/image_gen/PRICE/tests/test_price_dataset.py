from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_imgs  # noqa: E402
import price_dataset  # noqa: E402


class PriceDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        split = self.root / "dataset" / "data" / "test"
        (split / "query").mkdir(parents=True)
        (split / "output").mkdir()
        (split / "query" / "sample_1.jpg").write_bytes(b"query")
        (split / "output" / "sample_1.jpg").write_bytes(b"reference")
        metadata = {
            "id": "sample_1",
            "query_file_name": "query/sample_1.jpg",
            "output_file_name": "output/sample_1.jpg",
            "lang": "move the object",
            "dataset": "example",
        }
        (split / "metadata.jsonl").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )

        self.predictions = self.root / "predictions"
        (self.predictions / "imgs").mkdir(parents=True)
        (self.predictions / "imgs" / "sample_1.png").write_bytes(b"prediction")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_loads_hf_imagefolder_metadata_and_matches_prediction(self) -> None:
        dataset_root = self.root / "dataset"
        manifest = price_dataset.load_manifest(dataset_root)
        self.assertEqual(list(manifest), ["sample_1"])

        samples = eval_imgs.load_samples(
            self.predictions,
            dataset_root=dataset_root,
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["id"], "sample_1")
        self.assertEqual(samples[0]["instruction"], "move the object")
        self.assertTrue(Path(samples[0]["input_image"]).is_file())
        self.assertTrue(Path(samples[0]["output_image"]).is_file())


if __name__ == "__main__":
    unittest.main()
