# pylint: skip-file
from dataclasses import Field, fields, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Tuple, Type, TypeVar

from google.cloud import bigquery
from more_itertools import first
from typing_inspect import get_args, is_optional_type


class timestamp(datetime):
    """Helper type to distinguish naive datetime from an absolute point in time in type hints."""


def unix_timestamp(t: datetime) -> int:
    return int(t.replace(tzinfo=timezone.utc).timestamp())


def escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def value_type(field_type: Type) -> Type:
    """If the field type is an `Optional[T], returns T. Otherwise returns T unchanged."""

    if is_optional_type(field_type):
        return get_args(field_type)[0]
    return field_type


def bigquery_type(field_type: Type) -> str:
    # pylint: disable=too-many-return-statements
    t = value_type(field_type)
    if issubclass(t, str):
        return "STRING"
    if issubclass(t, Decimal):
        return "NUMERIC"
    if issubclass(t, bool):
        return "BOOLEAN"
    if issubclass(t, int):
        return "INT64"
    if issubclass(t, float):
        return "FLOAT"
    if issubclass(t, timestamp):
        return "TIMESTAMP"
    if issubclass(t, datetime):
        return "DATETIME"
    if issubclass(t, date):
        return "DATE"
    if is_dataclass(field_type):
        struct_fields = [
            f"`{f.name}` {bigquery_type(f.type)}"
            for f in fields(field_type)  # fmt: skip
        ]
        return f"STRUCT<{', '.join(struct_fields)}>"

    raise ValueError(f'Unserializable type "{field_type}"')


def serialized_value(value: Any, field_type: Type) -> str:
    # pylint: disable=too-many-return-statements
    if isinstance(value, str):
        return f"'{escape(value)}'"
    if isinstance(value, Decimal):
        return f"NUMERIC '{value}'"
    if isinstance(value, bool):
        return f"{str(value).upper()}"
    if isinstance(value, timestamp):
        return f"TIMESTAMP '{value.isoformat()}'"
    if isinstance(value, datetime):
        return f"DATETIME '{value.isoformat()}'"
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, dict):
        return "STRUCT(" + ", ".join(serialized_value(v, ...) + " AS " + k for k, v in value.items()) + ")"
    if isinstance(value, list):
        return "[" + ", ".join([serialized_value(v, ...) for v in value]) + "]"
    if is_dataclass(value):
        struct_values = [
            serialized_value(v, f.type) + " AS " + f.name
            for f, v in fields_and_values(value)  # fmt: skip
        ]
        return "STRUCT(" + ", ".join(struct_values) + ")"
    return str(value)


def null_value(field_type: Type) -> str:
    # pylint: disable=too-many-return-statements
    if issubclass(field_type, str):
        return "CAST(NULL AS STRING)"
    if issubclass(field_type, Decimal):
        return "CAST(NULL AS NUMERIC)"
    if issubclass(field_type, bool):
        return "CAST(NULL AS BOOLEAN)"
    if issubclass(field_type, int):
        return "CAST(NULL AS INT64)"
    if issubclass(field_type, float):
        return "CAST(NULL AS FLOAT)"
    if issubclass(field_type, timestamp):
        return "CAST(NULL AS TIMESTAMP)"
    if issubclass(field_type, datetime):
        return "CAST(NULL AS DATETIME)"
    if issubclass(field_type, date):
        return "CAST(NULL AS DATE)"
    if is_dataclass(field_type):
        bq_type = bigquery_type(field_type)
        return f"CAST(NULL AS {bq_type})"
    raise ValueError(f'Unserializable type "{field_type}"')


def serialize_field(field: Field, value: Any) -> str:
    vt = value_type(field.type)
    if value is None and is_optional_type(field.type):
        return f"{null_value(vt)} AS {field.name}"
    return f"{serialized_value(value, vt)} AS {field.name}"


def fields_and_values(obj: Any) -> Iterable[Tuple[Field, Any]]:
    assert is_dataclass(obj)

    for field in fields(obj):
        value = getattr(obj, field.name)
        yield field, value


def as_sql(obj: Any) -> str:
    from collections.abc import Iterable

    if is_dataclass(obj):
        data = ",\n  ".join(serialize_field(f, v) for f, v in fields_and_values(obj))
        return "SELECT\n  " + data
    elif isinstance(obj, Iterable):
        return "\n UNION ALL \n".join([f"({as_sql(s)}) " for s in obj])


def extract_name(obj: Any) -> str:
    from collections.abc import Iterable

    if is_dataclass(obj):
        return obj.__class__.__name__
    elif isinstance(obj, Iterable):
        return first([extract_name(s) for s in obj], obj.__class__.__name__)


def empty_sql(cls: Type) -> str:
    """Returns a query that generates an empty set of the correct type."""

    assert is_dataclass(cls)

    field_declarations = ",\n  ".join(f"{field.name} {bigquery_type(field.type)}" for field in fields(cls))
    return f"SELECT *\nFROM UNNEST(ARRAY<STRUCT<\n  {field_declarations}\n>>[])"


T = TypeVar("T")  # pylint: disable=invalid-name


def _map_types(field_type: Type[T], value: Any) -> T:
    if issubclass(field_type, timestamp):
        return timestamp(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            value.tzinfo,
        )
    return value


def from_query_result(cls: Type[T], row: bigquery.table.Row) -> T:
    params = {field.name: _map_types(field.type, getattr(row, field.name)) for field in fields(cls)}
    return cls(**params)
