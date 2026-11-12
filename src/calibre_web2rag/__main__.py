"""Command line: `calibre-web2rag` or `python -m calibre_web2rag`.

Reads the Calibre library once, upserts it into Qdrant and exits.
"""

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from types import FrameType

from dotenv import load_dotenv

from calibre_web2rag import __version__
from calibre_web2rag.config import ConfigError, Settings
from calibre_web2rag.ingest import ingest

logger = logging.getLogger("calibre_web2rag")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calibre-web2rag", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args(argv)

    # A local .env is a convenience for running outside Docker; variables
    # already set in the environment win.
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"calibre-web2rag: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # `docker stop` sends SIGTERM: finish the file being read, upsert, then stop.
    stop = threading.Event()

    def on_sigterm(signum: int, frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping after the current batch", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_sigterm)

    logger.info("calibre-web2rag %s starting", __version__)
    try:
        count = ingest(settings, stop=stop)
    except Exception:
        # Top level of a batch job: log the traceback and fail the run.
        logger.exception("Ingestion failed")
        return 1
    if stop.is_set():
        logger.info("Stopped early. Upserted %s chunks", count)
    else:
        logger.info("Ingest complete. Upserted %s chunks", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
