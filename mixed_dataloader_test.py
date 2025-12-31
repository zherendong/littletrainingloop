"""
Test script to demonstrate the deterministic MixedDataLoader.
"""

from typing import Any, Iterable
from training_basics import DataProvider
from language_model_dataloader import MixedDataLoader


class SimpleDataProvider(DataProvider[dict[str, Any]]):
    """Simple data provider that yields numbered items."""

    def __init__(self, name: str, count: int):
        self.name_prefix = name
        self.count = count

    def generate(self) -> Iterable[dict[str, Any]]:
        """Generate numbered items."""
        for i in range(self.count):
            yield {"source": self.name_prefix, "index": i}

    def get_name(self) -> str:
        return f"{self.name_prefix}DataProvider"


def test_mixed_dataloader_deterministic():
    """Test that MixedDataLoader produces deterministic results."""
    # Create three data providers
    loader_a = SimpleDataProvider("A", 10)
    loader_b = SimpleDataProvider("B", 10)
    loader_c = SimpleDataProvider("C", 10)

    # Mix them with weights [2, 1, 1]
    # This means we should get roughly 50% from A, 25% from B, 25% from C
    mixed = MixedDataLoader(
        data_loaders=[loader_a, loader_b, loader_c],
        weights=[2.0, 1.0, 1.0],
        name="TestMixed",
    )

    # Generate items twice and verify they're identical
    # Take first 60 items (2 full cycles through all loaders)
    items_first = []
    gen1 = iter(mixed.generate())
    for _ in range(60):
        items_first.append(next(gen1))

    items_second = []
    gen2 = iter(mixed.generate())
    for _ in range(60):
        items_second.append(next(gen2))

    print(f"Total items: {len(items_first)}")
    print("\nFirst 20 items from first run:")
    for i, item in enumerate(items_first[:20]):
        print(f"  {i}: {item}")

    print("\nFirst 20 items from second run:")
    for i, item in enumerate(items_second[:20]):
        print(f"  {i}: {item}")

    # Verify determinism
    assert items_first == items_second, "Results should be deterministic!"
    print("\n✓ Results are deterministic!")

    # Verify proportions
    count_a = sum(1 for item in items_first if item["source"] == "A")
    count_b = sum(1 for item in items_first if item["source"] == "B")
    count_c = sum(1 for item in items_first if item["source"] == "C")

    print(f"\nCounts: A={count_a}, B={count_b}, C={count_c}")
    print(
        f"Proportions: A={count_a/len(items_first):.2%}, B={count_b/len(items_first):.2%}, C={count_c/len(items_first):.2%}"
    )
    print(f"Expected: A=50%, B=25%, C=25%")

    # With weights [2, 1, 1], we should get roughly 2:1:1 ratio
    # Over 60 items, that's approximately 30:15:15
    assert abs(count_a - 30) <= 2, f"Expected ~30 items from A, got {count_a}"
    assert abs(count_b - 15) <= 2, f"Expected ~15 items from B, got {count_b}"
    assert abs(count_c - 15) <= 2, f"Expected ~15 items from C, got {count_c}"
    print("✓ Proportions are correct!")


def test_mixed_dataloader_with_different_lengths():
    """Test MixedDataLoader with iterators of different lengths - should restart."""
    # Create data providers with different lengths
    loader_a = SimpleDataProvider("A", 20)  # Longer
    loader_b = SimpleDataProvider("B", 5)  # Shorter

    # Mix with equal weights
    mixed = MixedDataLoader(
        data_loaders=[loader_a, loader_b],
        weights=[1.0, 1.0],
    )

    # Take 50 items - this should cause B to restart multiple times
    items = []
    gen = iter(mixed.generate())
    for _ in range(50):
        items.append(next(gen))

    print(f"\n\nTest with different lengths (with restart):")
    print(f"Total items taken: {len(items)}")

    count_a = sum(1 for item in items if item["source"] == "A")
    count_b = sum(1 for item in items if item["source"] == "B")

    print(f"Counts: A={count_a}, B={count_b}")

    # With equal weights, we should get roughly equal counts
    # B should have restarted multiple times
    print(f"B restarted approximately {count_b // 5} times")

    # Show the sequence
    print("\nSequence of sources (first 50):")
    sequence = "".join(item["source"] for item in items)
    print(f"  {sequence}")

    # Verify B items cycled through indices
    b_items = [item for item in items if item["source"] == "B"]
    b_indices = [item["index"] for item in b_items]
    print(f"\nB indices: {b_indices[:15]}...")  # Show first 15

    # Should see indices 0-4 repeating
    assert 0 in b_indices and 4 in b_indices
    print("✓ B dataloader restarted correctly!")


def test_weighted_distribution():
    """Test that weights are respected in the distribution."""
    loader_a = SimpleDataProvider("A", 100)
    loader_b = SimpleDataProvider("B", 100)

    # Mix with 3:1 ratio
    mixed = MixedDataLoader(
        data_loaders=[loader_a, loader_b],
        weights=[3.0, 1.0],
    )

    # Take 400 items to see the pattern over multiple cycles
    items = []
    gen = iter(mixed.generate())
    for _ in range(400):
        items.append(next(gen))

    print(f"\n\nTest weighted distribution (3:1):")

    # Look at first 40 items to see the pattern
    print("First 40 items:")
    sequence = "".join(item["source"] for item in items[:40])
    print(f"  {sequence}")

    # The pattern should be roughly AAABAAABAAAB...
    # Let's verify the overall distribution
    count_a = sum(1 for item in items if item["source"] == "A")
    count_b = sum(1 for item in items if item["source"] == "B")

    print(f"Total counts: A={count_a}, B={count_b}")
    print(f"Ratio: {count_a/count_b:.2f} (expected 3.0)")

    # With 3:1 weights, we should get roughly 300:100 ratio
    assert abs(count_a - 300) <= 5, f"Expected ~300 from A, got {count_a}"
    assert abs(count_b - 100) <= 5, f"Expected ~100 from B, got {count_b}"
    print("✓ Weighted distribution is correct!")
