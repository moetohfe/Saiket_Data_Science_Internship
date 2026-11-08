"""Unit tests for the data cleaning, encoding and segmentation logic."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import churn_rate, clean, load_raw  # noqa: E402
from task1_data_preparation import encode  # noqa: E402
from task3_segmentation import add_segments  # noqa: E402


class TestCleaning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_raw()
        cls.df = clean(cls.raw)

    def test_no_missing_values_after_cleaning(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_total_charges_is_numeric(self):
        self.assertTrue(pd.api.types.is_float_dtype(self.df["TotalCharges"]))

    def test_new_customers_have_zero_total_charges(self):
        new = self.df[self.df["tenure"] == 0]
        self.assertEqual(len(new), 11)
        self.assertTrue((new["TotalCharges"] == 0).all())

    def test_senior_citizen_is_yes_no(self):
        self.assertEqual(set(self.df["SeniorCitizen"]), {"Yes", "No"})

    def test_raw_data_unchanged(self):
        self.assertEqual(self.raw.shape, (7043, 21))


class TestEncoding(unittest.TestCase):
    def test_encoded_data_is_all_numeric(self):
        encoded = encode(clean(load_raw()))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(t) for t in encoded.dtypes))
        self.assertNotIn("customerID", encoded.columns)
        self.assertEqual(set(encoded["Churn"]), {0, 1})


class TestHelpers(unittest.TestCase):
    def test_churn_rate(self):
        self.assertAlmostEqual(churn_rate(pd.Series(["Yes", "No", "No", "No"])), 25.0)

    def test_overall_churn_rate(self):
        self.assertAlmostEqual(churn_rate(clean(load_raw())["Churn"]), 26.54, places=2)

    def test_segments_cover_every_customer(self):
        seg = add_segments(clean(load_raw()))
        self.assertEqual(int(seg["TenureSegment"].isna().sum()), 0)
        self.assertEqual(int(seg["SpendSegment"].isna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
