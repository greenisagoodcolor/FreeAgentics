"""Simple coverage test to verify coverage tool works."""

def add_numbers(a, b):
    """Add two numbers."""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers."""
    return a * b

def main():
    """Main function."""
    result1 = add_numbers(2, 3)
    result2 = multiply_numbers(4, 5)
    print(f"Results: {result1}, {result2}")
    return result1, result2

if __name__ == "__main__":
    main()
