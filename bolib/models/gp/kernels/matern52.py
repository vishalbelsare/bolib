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

import numpy as np

import bolib.models.gp.kernels.util

SQRT_5 = np.sqrt(5.0)


def stationary_function(sq_dist):
    """ It applies the Matern (v=5/2) kernel function
    element-wise to the distance matrix.

    .. math::
        k_{M52}(r)=(1+\dfrac{\sqrt{5}r}{l} +
        \dfrac{5r^2}{3l^2}) exp (-\dfrac{\sqrt{5}r}{l})

    :param sq_dist: Distance matrix
    :type sq_dist: np.matrix

    :return: Result matrix with kernel function applied element-wise.
    :rtype: np.matrix """
    dist = np.sqrt(sq_dist)
    return np.matrix(np.multiply(
        (1.0 + SQRT_5*dist + (5.0/3.0)*sq_dist), np.exp(-SQRT_5*dist)))


def kernel_function(mat_a, mat_b, lengthscale):
    """ Measures the distance matrix between solutions of A and B, and applies
    the kernel function element-wise to the distance matrix.
    
    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: Result matrix with kernel function applied element-wise.
    :rtype: np.matrix """
    sq_dist = bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale)
    return stationary_function(sq_dist)


def dk_dx(mat_a, mat_b, lengthscale):
    """ 
    Measures gradient of the kernel function in X.
    
    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: 3D array with the gradient of the kernel function in every
     dimension of X.
    :rtype: np.array """
    dr_dx = bolib.models.gp.kernels.util.dr_dx(
        mat_a, mat_b, lengthscale)
    sq_dist = np.array(bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale))
    dist = np.sqrt(sq_dist)
    grad_r2 = -(5.0/6.0)*np.exp(-SQRT_5*dist)*(1 + SQRT_5*dist)
    return grad_r2[:, :, np.newaxis] * dr_dx


def dk_dl(mat_a, mat_b, lengthscale):
    """ 
    Measures gradient of the kernel function in the length-scale
    hyper-parameter space.

    :param mat_a: List of solutions in lines and dimensions in columns.
    :type mat_a: np.matrix 
    :param mat_b: List of solutions in lines and dimensions in columns.
    :type mat_b: np.matrix
    :param lengthscale: Array of lenghtscale parameters. One per dimension
     in ARD case, only one element otherwise.  
    :type lengthscale: np.array
    :return: 3D array with the gradient of the kernel function in every
     dimension the length-scale hyper-parameter space.
    :rtype: np.array """
    dr_dl = bolib.models.gp.kernels.util.dr_dl(
        mat_a, mat_b, lengthscale)
    sq_dist = np.array(bolib.models.gp.kernels.util.sq_distance(
        mat_a, mat_b, lengthscale))
    dist = np.sqrt(sq_dist)
    grad_r2 = -(5.0/6.0)*np.exp(-SQRT_5*dist)*(1 + SQRT_5*dist)
    return grad_r2[:, :, np.newaxis] * dr_dl
