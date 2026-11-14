"""Small, testable helpers shared by the Streamlit application."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError


def decode_image_bytes(image_bytes, max_size_bytes, max_pixels):
    if len(image_bytes) > max_size_bytes:
        raise ValueError(
            f"Image is too large. Maximum upload size is "
            f"{max_size_bytes // (1024 * 1024)} MB."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.width * source.height > max_pixels:
                raise ValueError("Image dimensions are too large to process.")
            source.verify()

        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not read the uploaded image: {exc}") from exc
