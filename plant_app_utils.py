"""Small, testable helpers shared by the Streamlit application."""

from io import BytesIO
from datetime import datetime, timezone
import math
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


HISTORY_DEDUPE_COLUMNS = (
    "Date",
    "Username",
    "Image",
    "Prediction",
    "Confidence",
    "Risk",
)


def create_history_record(username, filename, prediction, confidence, risk, now=None):
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("History timestamps must include a timezone.")

    return {
        "Record ID": uuid4().hex,
        "Date": timestamp.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "Username": username,
        "Image": filename,
        "Prediction": prediction,
        "Confidence": f"{confidence:.2f}%",
        "Risk": risk,
    }


def deduplicate_history_records(records):
    unique_by_key = {}
    ordered_keys = []

    for record in records:
        record_id = record.get("Record ID")
        has_record_id = (
            record_id is not None
            and record_id != ""
            and not (
                isinstance(record_id, float)
                and math.isnan(record_id)
            )
        )
        if has_record_id:
            key = ("id", str(record_id))
        else:
            key = (
                "legacy",
                tuple(record.get(column) for column in HISTORY_DEDUPE_COLUMNS),
            )
        if key not in unique_by_key:
            ordered_keys.append(key)
        unique_by_key[key] = record

    return [unique_by_key[key] for key in ordered_keys]


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
