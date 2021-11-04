"""
Unit tests.
"""
import unittest
from contextlib import redirect_stdout
import io
from td3a_cpp_deep import check


class TestCheck(unittest.TestCase):

    def test_check(self):
        f = io.StringIO()
        with redirect_stdout(f):
            res = check(verbose=1)
        self.assertIsInstance(res, list)


if __name__ == '__main__':
    unittest.main()
