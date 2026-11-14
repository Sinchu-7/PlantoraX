import unittest
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from plant_app_utils import (
    create_history_record,
    decode_image_bytes,
    escape_pdf_text,
    deduplicate_history_records,
    get_model_image_size,
    validate_model_class_count,
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


if __name__ == "__main__":
    unittest.main()
