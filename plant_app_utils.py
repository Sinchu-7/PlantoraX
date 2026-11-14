"""Small, testable helpers shared by the Streamlit application."""

from io import BytesIO
from datetime import datetime, timezone
import base64
import csv
import hashlib
import html
import hmac
import math
import os
import re
import secrets
import tempfile
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image, UnidentifiedImageError


HISTORY_DEDUPE_COLUMNS = (
    "Date",
    "Username",
    "Image",
    "Prediction",
    "Confidence",
    "Risk",
)
MAX_PROFILE_PHOTO_BYTES = 5 * 1024 * 1024
MAX_PROFILE_PHOTO_PIXELS = 10_000_000
PROFILE_PHOTO_MAX_DIMENSION = 512
PASSWORD_HASH_ITERATIONS = 600_000
PASSWORD_MIN_LENGTH = 12


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


def migrate_history_csv(history_path):
    history_path = Path(history_path)
    if not history_path.is_file():
        return False

    with history_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return False
        records = list(reader)

    if "Record ID" not in fieldnames:
        fieldnames.insert(0, "Record ID")
        needs_migration = True
    else:
        needs_migration = any(not record.get("Record ID") for record in records)

    if not needs_migration:
        return False

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=history_path.parent,
            prefix=".prediction-history-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(
                temporary_file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for record in records:
                if not record.get("Record ID"):
                    record["Record ID"] = uuid4().hex
                writer.writerow(record)
        os.replace(temporary_path, history_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return True


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


def escape_pdf_text(value):
    return escape(str(value))


def sanitize_spreadsheet_value(value):
    if not isinstance(value, str):
        return value

    formula_candidate = value.lstrip(" \t\r\n\ufeff")
    if formula_candidate.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def sanitize_spreadsheet_dataframe(dataframe):
    safe_dataframe = dataframe.copy()
    for column in safe_dataframe.columns:
        if (
            pd.api.types.is_object_dtype(safe_dataframe[column].dtype)
            or pd.api.types.is_string_dtype(safe_dataframe[column].dtype)
        ):
            safe_dataframe[column] = safe_dataframe[column].map(
                sanitize_spreadsheet_value
            )
    return safe_dataframe


def escape_html_text(value):
    return html.escape(str(value), quote=True)


def dataframe_to_html(dataframe):
    return dataframe.to_html(
        index=False,
        escape=True,
        border=0,
        classes="app-dataframe",
    )


def validate_profile_fields(name, email, phone):
    name = name.strip()
    email = email.strip()
    phone = phone.strip()

    if not name or len(name) > 80:
        raise ValueError("Name must contain between 1 and 80 characters.")
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Enter a valid email address.")
    if len(email) > 254:
        raise ValueError("Email address must be 254 characters or fewer.")
    if phone and (
        len(phone) > 32
        or not re.fullmatch(r"[0-9+().\-\s]+", phone)
    ):
        raise ValueError("Enter a valid phone number.")

    return name, email, phone


def normalize_profile_photo(image_bytes):
    image = decode_image_bytes(
        image_bytes,
        max_size_bytes=MAX_PROFILE_PHOTO_BYTES,
        max_pixels=MAX_PROFILE_PHOTO_PIXELS,
    )
    image.thumbnail(
        (PROFILE_PHOTO_MAX_DIMENSION, PROFILE_PHOTO_MAX_DIMENSION)
    )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def hash_password(password, salt=None, iterations=PASSWORD_HASH_ITERATIONS):
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    if not 1 <= iterations <= 2_000_000:
        raise ValueError("Password hash iteration count is out of range.")

    salt = salt or secrets.token_bytes(16)
    if not isinstance(salt, bytes) or len(salt) < 16:
        raise ValueError("Password hash salt must contain at least 16 bytes.")
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password, encoded_hash):
    try:
        algorithm, iteration_text, salt_text, digest_text = encoded_hash.split("$")
        iterations = int(iteration_text)
        if algorithm != "pbkdf2_sha256" or not 1 <= iterations <= 2_000_000:
            return False
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected_digest = base64.urlsafe_b64decode(digest_text.encode("ascii"))
        actual_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(actual_digest, expected_digest)
    except (AttributeError, TypeError, ValueError, UnicodeEncodeError):
        return False


def validate_new_password(password):
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters long."
        )


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
