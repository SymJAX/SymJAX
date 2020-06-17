import doctest
import unittest

suite = unittest.TestSuite()
suite.addTest(doctest.DocTestSuite("symjax.base"))
suite.addTest(doctest.DocTestSuite("symjax.tensor.base"))
suite.addTest(doctest.DocTestSuite("symjax.tensor.control_flow"))
runner = unittest.TextTestRunner()
runner.run(suite)
