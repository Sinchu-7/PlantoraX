import unittest
from io import BytesIO

from PIL import Image

from plant_app_utils import decode_image_bytes


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


if __name__ == "__main__":
    unittest.main()
