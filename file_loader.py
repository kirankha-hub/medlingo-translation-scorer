"""Robust spreadsheet loader.

Detects the real file format from its content (magic bytes) instead of
trusting the filename extension. This handles the common case of an Excel
workbook saved with a ``.csv`` extension (or a CSV named ``.xlsx``), which
otherwise crashes ``pd.read_csv`` with errors like:

    'utf-8' codec can't decode byte 0xa9 in position 14: invalid start byte
"""

import io

import pandas as pd

# Magic-byte signatures.
_ZIP_SIG = b"PK\x03\x04"          # .xlsx / .xlsm (Office Open XML = ZIP)
_OLE_SIG = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .xls (OLE2)

# Encodings to try, in order, when reading a text (CSV/TSV) file.
_TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def _read_excel(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data))


def _read_text(data: bytes) -> pd.DataFrame:
    """Read CSV/TSV bytes, trying several encodings and auto-detecting the
    delimiter (comma vs tab vs semicolon)."""
    last_err = None
    for enc in _TEXT_ENCODINGS:
        try:
            text = data.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        # Sniff the delimiter from the first non-empty line.
        first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
        if "\t" in first_line:
            sep = "\t"
        elif first_line.count(";") > first_line.count(","):
            sep = ";"
        else:
            sep = ","
        return pd.read_csv(io.StringIO(text), sep=sep)
    # Could not decode as text with any known encoding.
    raise last_err if last_err else ValueError("Unable to decode file as text.")


def read_table(data: bytes, filename: str = "") -> pd.DataFrame:
    """Load a spreadsheet from raw bytes into a DataFrame.

    The format is chosen by content signature first, then by the filename
    extension as a hint, with a cross-fallback so a mislabeled file still
    loads.
    """
    if not data:
        raise ValueError("The file is empty.")

    head = data[:8]
    is_zip = head.startswith(_ZIP_SIG)
    is_ole = head.startswith(_OLE_SIG)

    # 1) Trust the content signature when it clearly says "Excel".
    if is_zip or is_ole:
        return _read_excel(data)

    # 2) Otherwise it's almost certainly text. Try text first...
    try:
        return _read_text(data)
    except Exception as text_err:
        # ...but fall back to Excel in case of an unusual/older binary format.
        try:
            return _read_excel(data)
        except Exception:
            raise text_err
