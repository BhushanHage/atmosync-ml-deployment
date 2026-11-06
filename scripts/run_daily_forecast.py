
import logging
import os
import sys

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

log = logging.getLogger("atmosync-daily-forecast")


def main():
    refresh_url = os.getenv("ATMOSYNC_REFRESH_URL")
    refresh_token = os.getenv("ATMOSYNC_REFRESH_TOKEN")

    if not refresh_url or not refresh_token:
        log.error("Missing forecast refresh environment variables.")
        sys.exit(1)

    try:
        response = requests.post(
            refresh_url,
            headers={
                "X-Atmosync-Refresh-Token": refresh_token,
            },
            timeout=180,
        )

        response.raise_for_status()
        result = response.json()

        generated = result.get("generated_count", 0)
        saved = result.get("saved_count", 0)
        errors = result.get("errors", [])

        log.info("Generated forecasts: %s", generated)
        log.info("Saved forecasts: %s", saved)
        log.info(
            "Forecast period: %s to %s",
            result.get("forecast_start_date"),
            result.get("forecast_end_date"),
        )

        if errors:
            log.error("Station errors: %s", errors)

        if generated != 63 or saved != 63 or errors:
            log.error("Forecast run incomplete.")
            sys.exit(1)

        log.info("Daily ATMOSYNC forecast completed successfully.")

    except requests.RequestException as exc:
        log.exception("Forecast refresh request failed: %s", exc)
        sys.exit(1)

    except (ValueError, TypeError) as exc:
        log.exception("Invalid refresh response: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
