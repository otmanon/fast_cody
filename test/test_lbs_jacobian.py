from .context import fast_cody as fcd
from .context import unittest
from .context import numpy as np
class TestLBSJacobian(unittest.TestCase):
    def test_rest(self):
        X = np.random.rand(100, 3)

        W = np.ones((100,1))

        J = fcd.lbs_jacobian(X,W)

        T0 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
        p0 = T0.flatten()[:, None]

        x_text = J @ p0

        X_test = x_text.reshape((100, 3), order='F')

        self.assertTrue(np.isclose(X_test, X).all)



if __name__ == '__main__':
    unittest.main()