import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

DATA_PATH = Path("data/spider/train_spider.json")


def main() -> None:
    examples = json.loads(DATA_PATH.read_text())
    counts: Counter = Counter(ex["db_id"] for ex in examples)
    sorted_dbs: List[Tuple[str, int]] = counts.most_common()

    print(f"\n{'Database':<40} {'Count':>6} {'Cumulative':>12}")
    print("-" * 60)
    cumulative = 0
    for db_id, count in sorted_dbs:
        cumulative += count
        print(f"{db_id:<40} {count:>6} {cumulative:>12}")

    all_counts = sorted(counts.values())
    n = len(all_counts)
    print(
        f"\nDistribution -- min: {all_counts[0]}, "
        f"p25: {all_counts[n // 4]}, "
        f"median: {all_counts[n // 2]}, "
        f"p75: {all_counts[3 * n // 4]}, "
        f"max: {all_counts[-1]}"
    )

    print("\nDatabases with at least N examples (as standalone strata):")
    for threshold in (50, 100, 150, 200):
        qualifying = [(db, c) for db, c in sorted_dbs if c >= threshold]
        total_examples = sum(c for _, c in qualifying)
        budget_for_100 = 100 * len(qualifying)
        print(
            f"  N={threshold:>3}: {len(qualifying):>3} databases, "
            f"{total_examples:>5} total examples, "
            f"budget for 100 labels each: {budget_for_100}"
        )


if __name__ == "__main__":
    main()
