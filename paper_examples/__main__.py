"""Run individual worked examples or all nine in manuscript order."""
import argparse
from importlib import import_module
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?", choices=[str(i) for i in range(1, 10)] + ["all"],
                        default="all", help="A manuscript example number (default: all)")
    args = parser.parse_args()
    numbers = range(1, 10) if args.example == "all" else (int(args.example),)
    for number in numbers:
        module = import_module(f"paper_examples.example{number:02}")
        print(json.dumps({"example": number, "result": module.run()}, indent=2))


if __name__ == "__main__":
    main()
