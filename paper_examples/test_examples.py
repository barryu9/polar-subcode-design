"""Run once per numbered example; detailed arithmetic checks live at their steps."""
from importlib import import_module
import unittest


class PaperExamples(unittest.TestCase):
    def test_nine_examples(self):
        for number in range(1, 10):
            with self.subTest(example=number):
                module = import_module(f"paper_examples.example{number:02}")
                self.assertTrue(module.run())


if __name__ == "__main__":
    unittest.main()
