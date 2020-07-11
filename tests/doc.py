import doctest
import unittest

suite = unittest.TestSuite()
suite.addTest(doctest.DocTestSuite("symjax.base"))
suite.addTest(doctest.DocTestSuite("symjax.tensor.base"))
suite.addTest(doctest.DocTestSuite("symjax.tensor.control_flow"))
suite.addTest(doctest.DocTestSuite("symjax.nn.schedules"))


runner = unittest.TextTestRunner()
runner.run(suite)
