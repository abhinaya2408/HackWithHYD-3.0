"""Build PulseMind's demo dataset, or import the real Kaggle review dataset.

Two subcommands:

    # Deterministic, self-contained demo dataset (committed to the repo)
    python data/prepare_dataset.py demo

    # Transform real Kaggle TSV files into PulseMind's schema
    python data/prepare_dataset.py kaggle --raw data/raw

Why both? The Kaggle "Amazon US Customer Reviews Dataset"
(https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)
is real customer feedback, but it is ~tens of GB of TSVs whose review dates stop in
2015. The PulseMind demo needs a *longitudinal* Jan -> Feb -> Mar story so a product
change can be linked to an outcome. So:

* `demo` writes a small labelled synthetic dataset with the exact Jan/Feb/Mar
  timeline the demo story requires. No real customer identities are fabricated --
  customer ids are anonymous placeholders (cust_0001, ...).
* `kaggle` extracts the reviews that actually talk about the product flows
  PulseMind reasons about (checkout, payments, delivery, ...), maps them into the
  PulseMind schema, and deterministically re-distributes them across the same demo
  timeline so you can run the whole demo on real customer words.

Themes are *topics*; a topic can appear in negative or positive feedback. Complaint
share is always computed over negative feedback only.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    PRODUCT_AREAS,
    THEMES,
    get_settings,
)

FEEDBACK_COLUMNS = [
    "feedback_id",
    "date",
    "customer_id",
    "channel",
    "product_area",
    "feedback_text",
    "rating",
    "sentiment",
    "theme",
]

PRODUCT_CHANGE_COLUMNS = [
    "change_id",
    "date",
    "change_name",
    "product_area",
    "problem_targeted",
    "expected_outcome",
]

CHANNELS = ["mobile_app", "web", "call_center", "email", "in_app_survey"]

# ---------------------------------------------------------------------------
# Theme definitions: which product area a topic belongs to, and realistic
# customer wording for it. These read like the short, blunt lines you find in
# real review datasets, which is what the Kaggle importer feeds in as well.
# ---------------------------------------------------------------------------
THEME_TEMPLATES: dict[str, dict[str, object]] = {
    "checkout_friction": {
        "product_area": "Checkout",
        "negative": [
            "Checkout is a nightmare - six screens and it kept resetting my saved address. I gave up twice.",
            "Why do I have to re-enter my card details every single time at checkout? Takes forever.",
            "Too many steps to buy one item. The checkout flow is unnecessarily complicated.",
            "Checkout page hangs on mobile and then abandons my cart. Very annoying.",
            "The coupon field at checkout is confusing and it emptied my basket when I applied it.",
            "I lost my whole cart at the payment step. Had to start over from scratch.",
            "Checkout asked me to verify my address three times before it would accept it.",
            "Could not check out as a guest, forced to create an account first. Frustrating.",
        ],
        "positive": [
            "Checkout V2 is so much faster, I was done in about thirty seconds.",
            "One-tap checkout is a huge improvement, order placed without any friction.",
            "Checkout finally remembers my address and card. Much smoother than before.",
        ],
    },
    "payment_reliability": {
        "product_area": "Payments",
        "negative": [
            "UPI payment failed but the money was debited from my account. Still not refunded.",
            "My UPI transaction has been pending for three days and the order is not confirmed.",
            "Payment keeps failing at the last step with a generic error, no useful message.",
            "Card declined twice and then UPI failed too. Could not complete the purchase.",
            "Paid through UPI, got a success screen, but no order was ever created.",
            "The app showed payment failed, so I paid again, and now I have been charged twice.",
            "UPI collect request just times out. Tried four times with two different banks.",
            "Wallet balance was deducted but the order failed. Support has not returned the money.",
        ],
        "positive": [
            "UPI payment went through instantly this time, no retry needed.",
            "Saved card worked first try, payment confirmation came straight away.",
        ],
    },
    "delivery_delay": {
        "product_area": "Delivery",
        "negative": [
            "Delivery is four days late and the tracking page has not updated at all.",
            "Ordered a week ago and it still has not shipped. No update from anyone.",
            "Package arrived late and the outer box was crushed.",
            "Promised next-day delivery, it turned up five days later.",
            "Tracking says delivered but nothing has reached me.",
        ],
        "positive": [
            "Arrived a day early and the packaging was in perfect condition.",
            "Delivery was quick and the tracking updates were accurate the whole way.",
        ],
    },
    "search_relevance": {
        "product_area": "Search",
        "negative": [
            "Search results show completely unrelated products, I cannot find what I need.",
            "Filters do not apply properly, search keeps returning out-of-stock items.",
            "Searching for an exact model number returns nothing useful.",
            "Search returns the same sponsored items no matter what I type.",
            "Sorting by price resets the moment I open a product.",
        ],
        "positive": [
            "Search found the exact model I wanted on the first try.",
        ],
    },
    "return_process": {
        "product_area": "Returns",
        "negative": [
            "Return pickup was rescheduled three times without informing me.",
            "Refund for my return is still pending after two weeks.",
            "Return flow in the app is confusing and no return label was generated.",
            "Had to call support just to start a return, the app would not let me.",
        ],
        "positive": [
            "Return was picked up on time and the refund hit my account in two days.",
        ],
    },
    "app_crash": {
        "product_area": "App Performance",
        "negative": [
            "The app crashes every time I open the orders tab.",
            "Latest update made the app extremely slow on my phone.",
            "App freezes during checkout and I have to force close it.",
            "Login screen gets stuck on a blank white page after the update.",
        ],
        "positive": [
            "The app feels noticeably faster after the latest update.",
        ],
    },
    "support_response": {
        "product_area": "Support",
        "negative": [
            "Support chat kept me waiting forty minutes and then closed without a resolution.",
            "No reply to my email for five days now.",
            "The agent disconnected twice and I had to explain everything again.",
            "Nobody could tell me why my payment failed. Just kept transferring me.",
        ],
        "positive": [
            "Support resolved my issue in one chat, very helpful agent.",
        ],
    },
}

# Negative-review rating distribution (1-2 stars dominate real complaint text).
NEGATIVE_RATINGS = [1, 1, 1, 2, 1, 2, 2, 1, 3]
POSITIVE_RATINGS = [5, 4, 5, 4, 5, 5]

# ---------------------------------------------------------------------------
# The longitudinal demo plan.
#
# Nov/Dec 2025 are baseline months: payment reliability is already a *known*
# (mild) topic, which is what later lets PulseMind answer "have we seen this
# before?". January spikes checkout friction. Checkout V2 ships 2026-02-15. By
# March, checkout complaints fall away and payment/UPI reliability becomes the
# emerging issue.
# ---------------------------------------------------------------------------
PERIOD_PLAN: dict[str, dict[str, object]] = {
    "2025-11": {
        "negative": {
            "checkout_friction": 5,
            "payment_reliability": 4,
            "delivery_delay": 7,
            "search_relevance": 4,
            "return_process": 3,
            "app_crash": 3,
            "support_response": 4,
        },
        "positive": 3,
    },
    "2025-12": {
        "negative": {
            "checkout_friction": 6,
            "payment_reliability": 4,
            "delivery_delay": 9,
            "search_relevance": 4,
            "return_process": 3,
            "app_crash": 3,
            "support_response": 4,
        },
        "positive": 3,
    },
    "2026-01": {
        "negative": {
            "checkout_friction": 18,
            "payment_reliability": 5,
            "delivery_delay": 7,
            "search_relevance": 4,
            "return_process": 3,
            "app_crash": 4,
            "support_response": 5,
        },
        "positive": 3,
    },
    "2026-02": {
        # Checkout V2 lands mid-month, so checkout complaints are front-loaded.
        "negative": {
            "checkout_friction": 12,
            "payment_reliability": 6,
            "delivery_delay": 7,
            "search_relevance": 4,
            "return_process": 4,
            "app_crash": 4,
            "support_response": 5,
        },
        "positive": 3,
    },
    "2026-03": {
        "negative": {
            "checkout_friction": 7,
            "payment_reliability": 16,
            "delivery_delay": 6,
            "search_relevance": 4,
            "return_process": 4,
            "app_crash": 5,
            "support_response": 6,
        },
        "positive": 4,
    },
}

# Theme -> (fraction of that theme's rows placed before the 15th, otherwise after)
EARLY_WEIGHTS: dict[str, float] = {
    "checkout_friction": 0.68,  # the problem the change targets: front-loaded
    "payment_reliability": 0.5,
}


@dataclass
class FeedbackRow:
    feedback_id: str
    date: str
    customer_id: str
    channel: str
    product_area: str
    feedback_text: str
    rating: int
    sentiment: str
    theme: str


def _sentiment_from_rating(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def _period_days(period: str) -> tuple[date, date]:
    year, month = (int(p) for p in period.split("-"))
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def _pick_date(rng: random.Random, period: str, theme: str) -> date:
    start, end = _period_days(period)
    midpoint = date(start.year, start.month, 15)
    early = rng.random() < EARLY_WEIGHTS.get(theme, 0.5)
    if early:
        return start + timedelta(days=rng.randint(0, (midpoint - start).days))
    return midpoint + timedelta(days=rng.randint(0, (end - midpoint).days))


def build_demo_feedback(seed: int = 42) -> list[FeedbackRow]:
    """Generate the deterministic synthetic demo dataset."""
    rng = random.Random(seed)
    rows: list[FeedbackRow] = []
    counter = 1

    for period, plan in PERIOD_PLAN.items():
        negative = plan["negative"]
        assert isinstance(negative, dict)
        for theme, count in negative.items():
            templates = THEME_TEMPLATES[theme]["negative"]
            area = str(THEME_TEMPLATES[theme]["product_area"])
            assert isinstance(templates, list)
            for i in range(int(count)):
                text = templates[i % len(templates)]
                rating = rng.choice(NEGATIVE_RATINGS)
                day = _pick_date(rng, period, theme)
                rows.append(
                    FeedbackRow(
                        feedback_id=f"FB-{counter:05d}",
                        date=day.isoformat(),
                        customer_id=f"cust_{rng.randint(1000, 9999)}",
                        channel=rng.choice(CHANNELS),
                        product_area=area,
                        feedback_text=text,
                        rating=rating,
                        sentiment="negative",
                        theme=theme,
                    )
                )
                counter += 1

        # A handful of positive/neutral rows so sentiment math is not monotone.
        for i in range(int(plan["positive"])):
            theme = THEMES[i % len(THEMES)]
            templates = THEME_TEMPLATES[theme]["positive"]
            area = str(THEME_TEMPLATES[theme]["product_area"])
            assert isinstance(templates, list)
            text = templates[i % len(templates)]
            rating = rng.choice(POSITIVE_RATINGS)
            start, end = _period_days(period)
            day = start + timedelta(days=rng.randint(0, (end - start).days))
            rows.append(
                FeedbackRow(
                    feedback_id=f"FB-{counter:05d}",
                    date=day.isoformat(),
                    customer_id=f"cust_{rng.randint(1000, 9999)}",
                    channel=rng.choice(CHANNELS),
                    product_area=area,
                    feedback_text=text,
                    rating=rating,
                    sentiment=_sentiment_from_rating(rating),
                    theme=theme,
                )
            )
            counter += 1

    rows.sort(key=lambda r: (r.date, r.feedback_id))
    # Re-number after sorting so ids follow chronological order.
    for idx, row in enumerate(rows, start=1):
        row.feedback_id = f"FB-{idx:05d}"
    return rows


def build_demo_product_changes() -> list[dict[str, str]]:
    """The one product change the demo story needs the PM to have recorded."""
    return [
        {
            "change_id": "CHG-001",
            "date": "2026-02-15",
            "change_name": "Checkout V2",
            "product_area": "Checkout",
            "problem_targeted": (
                "Checkout friction: multi-step flow, saved address and card being "
                "reset, confusing coupon and guest checkout"
            ),
            "expected_outcome": (
                "Reduce checkout-related complaints and cart abandonment"
            ),
        }
    ]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {len(rows):>4} rows -> {path}")


# ---------------------------------------------------------------------------
# Kaggle import
# ---------------------------------------------------------------------------

# Keyword -> (product_area, theme). First match wins, so the most specific
# payment/checkout keywords are checked before the generic ones.
AREA_KEYWORDS: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("upi", "payment failed", "payment declined", "money was debited",
         "refund not received", "charged twice", "transaction failed", "otp"),
        "Payments",
        "payment_reliability",
    ),
    (
        ("checkout", "check out", "cart", "coupon", "promo code", "basket",
         "place the order", "ordering process"),
        "Checkout",
        "checkout_friction",
    ),
    (
        ("delivery", "shipping", "shipped", "arrived late", "tracking", "parcel",
         "package"),
        "Delivery",
        "delivery_delay",
    ),
    (
        ("return", "refund", "exchange", "send it back"),
        "Returns",
        "return_process",
    ),
    (
        ("search", "filter", "find what", "search results"),
        "Search",
        "search_relevance",
    ),
    (
        ("app crash", "crashes", "app freezes", "app is slow", "update broke",
         "app keeps"),
        "App Performance",
        "app_crash",
    ),
    (
        ("customer service", "customer support", "no one replied", "support agent",
         "contact support", "customer care"),
        "Support",
        "support_response",
    ),
]

# How each theme's real rows are spread over the demo timeline. Checkout text is
# mostly mapped to the January problem window so the Checkout V2 outcome story
# still has real before/after evidence, while payment text is weighted towards
# March so the emerging-issue story has real evidence too.
PERIOD_WEIGHTS: dict[str, dict[str, float]] = {
    "checkout_friction": {"2026-01": 0.62, "2026-02": 0.26, "2026-03": 0.12},
    "payment_reliability": {"2026-01": 0.22, "2026-02": 0.30, "2026-03": 0.48},
    "delivery_delay": {"2026-01": 0.30, "2026-02": 0.34, "2026-03": 0.36},
    "return_process": {"2026-01": 0.30, "2026-02": 0.34, "2026-03": 0.36},
    "search_relevance": {"2026-01": 0.30, "2026-02": 0.34, "2026-03": 0.36},
    "app_crash": {"2026-01": 0.30, "2026-02": 0.34, "2026-03": 0.36},
    "support_response": {"2026-01": 0.30, "2026-02": 0.34, "2026-03": 0.36},
}


def classify_review(text: str) -> tuple[str, str] | None:
    """Map real review text onto PulseMind's (product_area, theme) vocabulary."""
    lowered = text.lower()
    for keywords, area, theme in AREA_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return area, theme
    return None


def import_kaggle(
    raw_dir: Path,
    out_dir: Path,
    seed: int = 7,
    max_per_theme_per_period: int = 14,
    scan_rows_per_file: int = 200_000,
) -> None:
    """Transform Kaggle amazon_reviews_us_*.tsv files into PulseMind's schema."""
    files = sorted(raw_dir.glob("*.tsv"))
    if not files:
        raise SystemExit(
            f"No .tsv files found in {raw_dir}.\n"
            "Download the Kaggle dataset (see README) and extract the TSVs into "
            "that folder, then re-run this command."
        )

    print(f"Scanning {len(files)} TSV file(s) from {raw_dir} ...")
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}

    for path in files:
        print(f"  reading {path.name} (up to {scan_rows_per_file:,} rows)")
        try:
            frame = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                quoting=csv.QUOTE_NONE,
                on_bad_lines="skip",
                nrows=scan_rows_per_file,
                engine="python",
            )
        except Exception as exc:  # pragma: no cover - depends on user's files
            print(f"    skipped ({exc})")
            continue

        text_col = "review_body" if "review_body" in frame.columns else None
        if text_col is None:
            print("    skipped (no review_body column - is this the Kaggle TSV?)")
            continue

        headline = frame["review_headline"].fillna("") if "review_headline" in frame else ""
        body = frame[text_col].fillna("")
        ratings = frame["star_rating"] if "star_rating" in frame.columns else pd.Series(dtype=str)
        ids = frame["review_id"] if "review_id" in frame.columns else pd.Series(dtype=str)

        for idx in range(len(frame)):
            combined = f"{headline.iloc[idx] if len(headline) else ''} {body.iloc[idx]}".strip()
            if len(combined) < 25:
                continue
            hit = classify_review(combined)
            if hit is None:
                continue
            area, theme = hit
            try:
                rating = int(float(ratings.iloc[idx]))
            except (TypeError, ValueError):
                rating = 3
            buckets.setdefault((theme, area), []).append(
                {
                    "review_id": str(ids.iloc[idx]) if len(ids) else "",
                    "text": combined[:600],
                    "rating": str(rating),
                }
            )

    if not buckets:
        raise SystemExit(
            "No reviews matched PulseMind's flow vocabulary. Try a different "
            "category file (Electronics, Digital_Video_Download, PC ...)."
        )

    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    counter = 1

    for (theme, area), items in sorted(buckets.items()):
        weights = PERIOD_WEIGHTS.get(theme, {"2026-01": 0.34, "2026-02": 0.33, "2026-03": 0.33})
        rng.shuffle(items)
        # Take negatives first: the demo reasons about complaints.
        items.sort(key=lambda item: int(item["rating"]))
        cursor = 0
        for period, share in weights.items():
            want = max(1, int(round(max_per_theme_per_period * share)))
            taken = items[cursor : cursor + want]
            cursor += want
            if not taken:
                continue
            start, end = _period_days(period)
            span = (end - start).days
            for item in taken:
                rating = int(item["rating"])
                day = start + timedelta(days=rng.randint(0, span))
                rows.append(
                    {
                        "feedback_id": f"FB-{counter:05d}",
                        "date": day.isoformat(),
                        "customer_id": f"cust_{rng.randint(1000, 9999)}",
                        "channel": rng.choice(CHANNELS),
                        "product_area": area,
                        "feedback_text": item["text"],
                        "rating": rating,
                        "sentiment": _sentiment_from_rating(rating),
                        "theme": theme,
                    }
                )
                counter += 1

    rows.sort(key=lambda r: str(r["date"]))
    for idx, row in enumerate(rows, start=1):
        row["feedback_id"] = f"FB-{idx:05d}"

    print("Mapped real Kaggle reviews into the PulseMind demo timeline.")
    write_csv(out_dir / "feedback.csv", FEEDBACK_COLUMNS, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    settings = get_settings()
    default_out = settings.data_dir

    demo = sub.add_parser("demo", help="write the deterministic synthetic demo dataset")
    demo.add_argument("--out", type=Path, default=default_out)
    demo.add_argument("--seed", type=int, default=42)

    kaggle = sub.add_parser("kaggle", help="transform real Kaggle TSVs into PulseMind's schema")
    kaggle.add_argument("--raw", type=Path, default=default_out / "raw")
    kaggle.add_argument("--out", type=Path, default=default_out)
    kaggle.add_argument("--seed", type=int, default=7)
    kaggle.add_argument("--max-per-theme-per-period", type=int, default=14)
    kaggle.add_argument("--scan-rows-per-file", type=int, default=200_000)

    args = parser.parse_args()

    if args.command == "demo":
        print("Generating synthetic demo dataset (labelled as synthetic, no real identities).")
        rows = build_demo_feedback(seed=args.seed)
        write_csv(
            args.out / "feedback.csv",
            FEEDBACK_COLUMNS,
            [row.__dict__ for row in rows],
        )
        write_csv(
            args.out / "product_changes.csv",
            PRODUCT_CHANGE_COLUMNS,
            build_demo_product_changes(),
        )
        print(f"\nDone. {len(rows)} feedback rows across the Jan/Feb/Mar demo timeline.")
    else:
        import_kaggle(
            raw_dir=args.raw,
            out_dir=args.out,
            seed=args.seed,
            max_per_theme_per_period=args.max_per_theme_per_period,
            scan_rows_per_file=args.scan_rows_per_file,
        )


if __name__ == "__main__":
    main()
