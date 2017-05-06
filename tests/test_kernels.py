# -*- coding: utf-8 -*-
#
#    Copyright 2017 Ibai Roman
#
#    This file is part of BOlib.
#
#    BOlib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    BOlib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with BOlib. If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np

import bolib.models.gp.kernels.matern52 as matern52
import bolib.models.gp.kernels.matern32 as matern32
import bolib.models.gp.kernels.squared_exponential as squared_exponential
import bolib.models.gp.kernels.exponential as exponential
import bolib.models.gp.kernels.gamma_exponential15 as gamma_exponential15
import bolib.models.gp.kernels.rational_quadratic2 as rational_quadratic2


class KernelTest(unittest.TestCase):
    """ Kernel test set """

    def test_matern52(self):
        """ Test of Matern 52 kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = matern52
        res = np.matrix([[2.58954726e-04, 1.00000000e+00],
                         [2.13404908e-04, 5.23965288e-01],
                         [1.29098963e-04, 1.38655010e-01]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)
        res = np.array([[[-4.96239014e-04, 8.92516211e-07],
                         [-0.00000000e+00, -0.00000000e+00]],
                        [[-4.03451123e-04, 7.31572055e-05],
                         [-5.76395858e-03, 5.76395858e-01]],
                        [[-2.34821135e-04, 8.47379680e-05],
                         [-1.04174595e-03, 2.08349190e-01]]])
        np.testing.assert_allclose(kernel.dk_dx(
            mat_a, mat_b, lengthscale), res)

    def test_matern32(self):
        """ Test of Matern 32 kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = matern32
        res = np.matrix([[6.98540528e-04, 1.00000000e+00],
                         [5.96249695e-04, 4.83331188e-01],
                         [3.95437002e-04, 1.39726655e-01]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)
        res = np.array([[[-1.09608815e-03, 1.97138156e-06],
                         [-0.00000000e+00, -0.00000000e+00]],
                        [[-9.22117043e-04, 1.67206142e-04],
                         [-5.30717657e-03, 5.30717657e-01]],
                        [[-5.87018610e-04, 2.11832568e-04],
                         [-9.38992737e-04, 1.87798547e-01]]])
        np.testing.assert_allclose(
            kernel.dk_dx(mat_a, mat_b, lengthscale), res)

    def test_squared_exponential(self):
        """ Test of squared exponential kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = squared_exponential
        res = np.matrix([[1.93720391e-07, 1.00000000e+00],
                         [1.10031406e-07, 6.06500334e-01],
                         [2.43070355e-08, 1.35328517e-01]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)
        res = np.array([[[-1.07708538e-06, 1.93720391e-09],
                         [-0.00000000e+00, -0.00000000e+00]],
                        [[-6.12874933e-07, 1.11131720e-07],
                         [-6.06500334e-03, 6.06500334e-01]],
                        [[-1.35390188e-07, 4.88571414e-08],
                         [-1.35328517e-03, 2.70657033e-01]]])
        np.testing.assert_allclose(kernel.dk_dx(
            mat_a, mat_b, lengthscale), res)

    def test_exponential(self):
        """ Test of exponential kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = exponential
        res = np.matrix([[0.003848741786, 1.],
                         [0.003479626468, 0.367861048119],
                         [0.002680985743, 0.135331899918]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)
        res = np.array([[[-3.84873556e-03, 6.92218626e-06],
                         [0.00000000e+00, 0.00000000e+00]],
                        [[-3.42379444e-03, 6.20831668e-04],
                         [-3.67842656e-03, 3.67842656e-01]],
                        [[-2.52181241e-03, 9.10025664e-04],
                         [-6.76651042e-04, 1.35330208e-01]]])
        np.testing.assert_allclose(kernel.dk_dx(
            mat_a, mat_b, lengthscale), res)

    def test_gamma_exponential15(self):
        """ Test of gamma exponential 15 kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = gamma_exponential15
        res = np.matrix([[2.02423551e-06, 1.00000000e+00],
                         [1.41478883e-06, 3.67851852e-01],
                         [5.52001703e-07, 5.91026121e-02]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)
        res = np.array([[[-7.15960961e-06, 1.28769957e-08],
                         [0.00000000e+00, 0.00000000e+00]],
                        [[-4.96818782e-06, 9.00874273e-07],
                         [-5.51763984e-03, 5.51763984e-01]],
                        [[-1.89525832e-06, 6.83926250e-07],
                         [-6.26873949e-04, 1.25374790e-01]]])
        np.testing.assert_allclose(kernel.dk_dx(
            mat_a, mat_b, lengthscale), res)

    def test_rational_quadratic2(self):
        """ Test of rational quadratic 2 kernel """
        mat_a = np.matrix([[1.14, 14.1], [1.15, 13.1], [1.15, 12.1]])
        mat_b = np.matrix([[-4.42, 14.11], [1.14, 14.1]])
        lengthscale = np.matrix([[1, 1]])

        kernel = rational_quadratic2
        res = np.matrix([[0.013125873998, 1.],
                         [0.012314872589, 0.639974400768],
                         [0.010484417952, 0.249993750117]])
        np.testing.assert_allclose(kernel.kernel_function(mat_a, mat_b,
                                                          lengthscale), res)

        res = np.array([[[-8.36117162e-03, 1.50380785e-05],
                         [-0.00000000e+00, -0.00000000e+00]],
                        [[-7.61202278e-03, 1.38027702e-03],
                         [-5.11969281e-03, 5.11969281e-01]],
                        [[-5.97959380e-03, 2.15780674e-03],
                         [-1.24995313e-03, 2.49990625e-01]]])
        np.testing.assert_allclose(kernel.dk_dx(
            mat_a, mat_b, lengthscale), res)
