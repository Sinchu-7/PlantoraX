"""Small, testable helpers shared by the Streamlit application."""

from io import BytesIO

from PIL import Image, UnidentifiedImageError


def _single_shape(shape, name):
    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(f"Model must have exactly one {name}.")
        shape = shape[0]
    return tuple(shape)


def get_model_image_size(model):
    input_shape = _single_shape(
        getattr(model, "input_shape", None),
        "image input",
    )
    if len(input_shape) != 4:
        raise ValueError(
            "Model image input must have shape [batch, height, width, channels]."
        )

    height, width, channels = input_shape[1:]
    if height is None or width is None or channels != 3:
        raise ValueError(
            "Model must declare fixed image dimensions and three color channels."
        )
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError("Model image dimensions must be positive.")
    return int(height), int(width)


def validate_model_class_count(model, class_names):
    output_shape = _single_shape(
        getattr(model, "output_shape", None),
        "class-score output",
    )
    if len(output_shape) != 2 or output_shape[-1] is None:
        raise ValueError(
            "Model class-score output must have shape [batch, classes]."
        )
    if int(output_shape[-1]) != len(class_names):
        raise ValueError(
            f"Model returns {output_shape[-1]} classes, but "
            f"{len(class_names)} class labels are configured."
        )


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
