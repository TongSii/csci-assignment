#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:19:56 2023

@author: tsi2
"""
import pytest
from sparse_recommender import SparseMatrix



def test_add_value():
    sparse_matrix = SparseMatrix(3, 3)
    
    sparse_matrix.add_value(0, 0, 1)
    sparse_matrix.add_value(1, 1, 2)
    sparse_matrix.add_value(2, 2, 3)
    
    assert sparse_matrix.get_value(0, 0) == 1
    assert sparse_matrix.get_value(1, 1) == 2
    assert sparse_matrix.get_value(2, 2) == 3
    
def test_get_value():

    sparse_matrix = SparseMatrix(3, 3)  # Create a SparseMatrix

    sparse_matrix.add_value(0, 0, 1)
    sparse_matrix.add_value(1, 1, 2)
    sparse_matrix.add_value(2, 2, 3)

    assert sparse_matrix.get_value(0, 0) == 1
    assert sparse_matrix.get_value(1, 1) == 2
    assert sparse_matrix.get_value(2, 2) == 3
    assert sparse_matrix.get_value(0, 1) == 0  




def test_to_dense():
    sparse_matrix = SparseMatrix(3, 3)
    
    sparse_matrix.add_value(0, 0, 1)
    sparse_matrix.add_value(1, 1, 2)
    sparse_matrix.add_value(2, 2, 3)
    
    dense_matrix = sparse_matrix.to_dense() #convert to dense
    expected_dense_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    
    assert dense_matrix == expected_dense_matrix


def test_get_shape():
    sparse_matrix = SparseMatrix(3, 3)
    assert sparse_matrix.get_shape() == (3, 3)



def test_matrix_vector_multiply():
    sparse_matrix = SparseMatrix(3, 3)
    
    sparse_matrix.add_value(0, 0, 1)
    sparse_matrix.add_value(1, 1, 2)
    sparse_matrix.add_value(2, 2, 3)

    user_vector = [0.5, 1.0, 0.0] 
 

    with pytest.raises(ValueError):#add error handling
        
        recommendations = sparse_matrix.matrix_vector_multiply(user_vector)

    assert recommendations == [0.5, 2.0, 0.0]



def test_matrix_addition():
    sparse_matrix1 = SparseMatrix(3, 3)
    
    sparse_matrix1.add_value(0, 0, 1)
    sparse_matrix1.add_value(1, 1, 2)
    sparse_matrix1.add_value(2, 2, 3)

    sparse_matrix2 = SparseMatrix(3, 3)
    
    sparse_matrix2.add_value(0, 1, 1)
    sparse_matrix2.add_value(1, 2, 2)
    sparse_matrix2.add_value(2, 0, 3)

    result_matrix = sparse_matrix1.matrix_addition(sparse_matrix2)

    assert result_matrix.to_dense() == [[1, 1, 0], [0, 2, 2], [3, 0, 3]]

#Update your tests / Add tests for edge cases and error handling

#edge case : enpty matrix
def test_empty_matrix():
    sparse_matrix = SparseMatrix(0, 0)
    user_vector = []
    recommendations = sparse_matrix.matrix_vector_multiply(user_vector)
    assert recommendations == []

#error case, adding to "Multiplies"




