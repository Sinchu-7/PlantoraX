import unittest
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from plant_app_utils import (
    create_history_record,
    decode_image_bytes,
    escape_html_text,
    escape_pdf_text,
    deduplicate_history_records,
    get_model_image_size,
    hash_password,
    normalize_profile_photo,
    sanitize_spreadsheet_dataframe,
    sanitize_spreadsheet_value,
    validate_model_class_count,
    validate_new_password,
    validate_profile_fields,
    verify_password,
)


class DecodeImageBytesTests(unittest.TestCase):
    def make_image(self, size=(2, 2), image_format="PNG"):
        buffer = BytesIO()
        Image.new("RGB", size, color="green").save(buffer, format=image_format)
        return buffer.getvalue()

    def test_decodes_image_as_rgb(self):
        image = decode_image_bytes(self.make_image(), 1024, 100)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (2, 2))

    def test_rejects_upload_over_byte_limit(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            decode_image_bytes(self.make_image(), 1, 100)

    def test_rejects_image_over_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "dimensions are too large"):
            decode_image_bytes(self.make_image(size=(11, 10)), 1024, 100)

    def test_rejects_invalid_image_data(self):
        with self.assertRaisesRegex(ValueError, "Could not read"):
            decode_image_bytes(b"not an image", 1024, 100)


class ModelContractTests(unittest.TestCase):
    def test_reads_fixed_rgb_input_size(self):
        model = SimpleNamespace(input_shape=(None, 128, 160, 3))

        self.assertEqual(get_model_image_size(model), (128, 160))

    def test_rejects_dynamic_dimensions_and_non_rgb_inputs(self):
        for shape in (
            (None, None, 128, 3),
            (None, 128, 128, 1),
            (None, 128, 128),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    get_model_image_size(SimpleNamespace(input_shape=shape))

    def test_rejects_multiple_image_inputs(self):
        model = SimpleNamespace(input_shape=[(None, 128, 128, 3), (None, 8)])

        with self.assertRaisesRegex(ValueError, "exactly one"):
            get_model_image_size(model)

    def test_accepts_matching_class_count(self):
        validate_model_class_count(
            SimpleNamespace(output_shape=(None, 38)),
            ["class"] * 38,
        )

    def test_rejects_mismatched_class_count(self):
        with self.assertRaisesRegex(ValueError, "returns 37 classes"):
            validate_model_class_count(
                SimpleNamespace(output_shape=(None, 37)),
                ["class"] * 38,
            )

    def test_rejects_unknown_class_count(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_model_class_count(
                SimpleNamespace(output_shape=(None, None)),
                ["class"] * 38,
            )


class HistoryIntegrityTests(unittest.TestCase):
    def test_creates_timezone_aware_unique_record_ids(self):
        timestamp = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        first = create_history_record(
            "user", "leaf.jpg", "healthy", 95.125, "High Confidence", timestamp
        )
        second = create_history_record(
            "user", "leaf.jpg", "healthy", 95.125, "High Confidence", timestamp
        )

        self.assertEqual(first["Date"], "2025-01-02T03:04:05+00:00")
        self.assertNotEqual(first["Record ID"], second["Record ID"])

    def test_requires_timezone_aware_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            create_history_record(
            "user",
            "leaf.jpg",
            "healthy",
            95,
            "High Confidence",
            datetime(2025, 1, 2),
            )

    def test_deduplicates_csv_session_copy_but_keeps_same_second_predictions(self):
        first = create_history_record(
            "user", "leaf.jpg", "healthy", 95, "High Confidence"
        )
        same_record = dict(first)
        independent_record = create_history_record(
            "user", "leaf.jpg", "healthy", 95, "High Confidence"
        )

        result = deduplicate_history_records(
            [first, independent_record, same_record]
        )

        self.assertEqual(
            [record["Record ID"] for record in result],
            [first["Record ID"], independent_record["Record ID"]],
        )

    def test_deduplicates_legacy_records_without_record_ids(self):
        legacy = {
            "Date": "2025-01-02 03:04:05",
            "Username": "user",
            "Image": "leaf.jpg",
            "Prediction": "healthy",
            "Confidence": "95.00%",
            "Risk": "High Confidence",
        }

        self.assertEqual(
            deduplicate_history_records([legacy, dict(legacy)]),
            [legacy],
        )


class PdfTextTests(unittest.TestCase):
    def test_escapes_markup_and_ampersands(self):
        self.assertEqual(
            escape_pdf_text("<b>leaf & stem</b>"),
            "&lt;b&gt;leaf &amp; stem&lt;/b&gt;",
        )

    def test_converts_non_string_cell_values(self):
        self.assertEqual(escape_pdf_text(95.5), "95.5")


class SpreadsheetSafetyTests(unittest.TestCase):
    def test_prefixes_formula_candidates_after_leading_whitespace(self):
        for value in ("=1+1", " +SUM(A1:A2)", "\t@SUM(1,2)", "\ufeff-1+2"):
            with self.subTest(value=value):
                self.assertEqual(sanitize_spreadsheet_value(value), "'" + value)

    def test_leaves_non_formula_text_and_numeric_values_unchanged(self):
        self.assertEqual(sanitize_spreadsheet_value("leaf.jpg"), "leaf.jpg")
        self.assertEqual(sanitize_spreadsheet_value(-1), -1)

    def test_sanitizes_string_cells_without_mutating_source(self):
        source = pd.DataFrame(
            {"Image": ["=HYPERLINK(\"https://example.com\")"], "Confidence": [95.5]}
        )

        result = sanitize_spreadsheet_dataframe(source)

        self.assertEqual(result.loc[0, "Image"], "'=HYPERLINK(\"https://example.com\")")
        self.assertEqual(result.loc[0, "Confidence"], 95.5)
        self.assertEqual(source.loc[0, "Image"], '=HYPERLINK("https://example.com")')


class ProfileSafetyTests(unittest.TestCase):
    def test_escapes_profile_html(self):
        self.assertEqual(
            escape_html_text('<img src=x onerror="alert(1)">'),
            "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;",
        )

    def test_normalizes_and_downscales_valid_profile_photo(self):
        photo = normalize_profile_photo(
            DecodeImageBytesTests().make_image(size=(600, 300))
        )
        with Image.open(BytesIO(photo)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (512, 256))

    def test_rejects_oversized_profile_photo(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            normalize_profile_photo(b"x" * (5 * 1024 * 1024 + 1))

    def test_validates_and_normalizes_profile_fields(self):
        self.assertEqual(
            validate_profile_fields("  Green Thumb  ", "user@example.com", "+1 (555) 010-1234"),
            ("Green Thumb", "user@example.com", "+1 (555) 010-1234"),
        )

    def test_rejects_invalid_profile_fields(self):
        invalid_profiles = [
            ("", "user@example.com", ""),
            ("a" * 81, "", ""),
            ("Name", "not-an-email", ""),
            ("Name", "", "555-0101<script>"),
        ]
        for profile in invalid_profiles:
            with self.subTest(profile=profile):
                with self.assertRaises(ValueError):
                    validate_profile_fields(*profile)


class PasswordHashTests(unittest.TestCase):
    def test_hashes_with_unique_salts_and_verifies_password(self):
        first_hash = hash_password("a sufficiently long password", iterations=1000)
        second_hash = hash_password("a sufficiently long password", iterations=1000)

        self.assertNotEqual(first_hash, second_hash)
        self.assertTrue(verify_password("a sufficiently long password", first_hash))
        self.assertFalse(verify_password("wrong password", first_hash))

    def test_rejects_malformed_password_hashes(self):
        self.assertFalse(verify_password("password", "not-a-password-hash"))
        self.assertFalse(
            verify_password(
                "password",
                "pbkdf2_sha256$999999999$YWJjZGVmZ2hpamtsbW5vcA==$YWJj",
            )
        )

    def test_enforces_registration_password_length(self):
        with self.assertRaisesRegex(ValueError, "at least 12"):
            validate_new_password("short")
        validate_new_password("long enough password")


if __name__ == "__main__":
    unittest.main()
