from __future__ import annotations

import logging
import sys

from calibre_web2rag.config import load_settings
from calibre_web2rag.ingest import ingest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        settings = load_settings()
        count = ingest(settings)
        logging.info("Ingest complete. Upserted %s chunks", count)
    except Exception as exc:
        logging.exception("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    sys.exit(main())
