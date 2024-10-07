# NumPy Cheat Sheet

## Table of Contents
1. [Introduction](#introduction)
2. [Array Creation](#array-creation)
3. [Array Operations](#array-operations)
4. [Array Indexing and Slicing](#array-indexing-and-slicing)
5. [Array Shape Manipulation](#array-shape-manipulation)
6. [Mathematical Operations](#mathematical-operations)
7. [Statistical Operations](#statistical-operations)
8. [Linear Algebra](#linear-algebra)
9. [Random Number Generation](#random-number-generation)
10. [File I/O](#file-io)
11. [Flashcards](#flashcards)
12. [Additional Resources](#additional-resources)

## Introduction

NumPy is a powerful library for numerical computing in Python. This cheat sheet provides a quick reference for common NumPy operations and functions.
## Installation

```bash
pip install numpy
```



```python
import numpy as np
```

## Array Creation

```python
a = np.array([1, 2, 3])  # 1D array
b = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
c = np.zeros((3, 3))  # 3x3 array of zeros
d = np.ones((2, 2))  # 2x2 array of ones
e = np.eye(3)  # 3x3 identity matrix
f = np.arange(0, 10, 2)  # Array from 0 to 8 with step 2
g = np.linspace(0, 1, 5)  # 5 evenly spaced numbers from 0 to 1
```

## Array Operations

```python
a + b  # Element-wise addition
a * b  # Element-wise multiplication
a.dot(b)  # Matrix multiplication
np.exp(a)  # Exponential of each element
np.sqrt(a)  # Square root of each element
np.sin(a)  # Sine of each element
```

## Array Indexing and Slicing

```python
a[0]  # First element
b[0, 1]  # Element at first row, second column
a[1:3]  # Slice from index 1 to 2
b[:, 1]  # Second column
a[a > 2]  # Boolean indexing
```

## Array Shape Manipulation

```python
a.reshape(3, 1)  # Reshape to 3x1 array
b.T  # Transpose
c.ravel()  # Flatten array
np.vstack((a, b))  # Stack vertically
np.hstack((a, b))  # Stack horizontally
```

## Mathematical Operations

```python
np.add(a, b)  # Element-wise addition
np.subtract(a, b)  # Element-wise subtraction
np.multiply(a, b)  # Element-wise multiplication
np.divide(a, b)  # Element-wise division
np.power(a, 2)  # Element-wise exponentiation
```

## Statistical Operations

```python
np.mean(a)  # Mean
np.median(a)  # Median
np.std(a)  # Standard deviation
np.var(a)  # Variance
np.min(a)  # Minimum
np.max(a)  # Maximum
np.argmin(a)  # Index of minimum
np.argmax(a)  # Index of maximum
```

## Linear Algebra

```python
np.linalg.inv(a)  # Inverse of a matrix
np.linalg.det(a)  # Determinant of a matrix
np.linalg.eig(a)  # Eigenvalues and eigenvectors
np.linalg.solve(a, b)  # Solve linear system Ax = b

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.dot(a, b)                       # Matrix multiplication
np.linalg.inv(a)                   # Inverse of matrix
np.linalg.det(a)                   # Determinant of matrix
np.linalg.eig(a)                   # Eigenvalues and eigenvectors
np.transpose(a)                    # Transpose of matrix


```

## Sorting, Searching, and Counting

```python
arr = np.array([3, 1, 2, 4])

np.sort(arr)                       # Sort array
np.argsort(arr)                    # Indices of the sorted elements
np.argmax(arr)                     # Index of maximum value
np.argmin(arr)                     # Index of minimum value
np.where(arr > 2)                  # Indices where condition is true
np.count_nonzero(arr)              # Count non-zero elements
```

## Random Number Generation

```python
np.random.rand(3, 3)  # 3x3 array of random floats
np.random.randn(3, 3)  # 3x3 array from standard normal distribution
np.random.randint(0, 10, (3, 3))  # 3x3 array of random integers from 0 to 9
```

## File I/O

```python
np.save('array.npy', a)  # Save array to .npy file
b = np.load('array.npy')  # Load array from .npy file
np.savetxt('array.txt', a)  # Save array to text file
c = np.loadtxt('array.txt')  # Load array from text file
```

## Flashcards

1. Q: How to create a 3x3 identity matrix?
   A: `np.eye(3)`

2. Q: How to compute element-wise exponential of an array?
   A: `np.exp(array)`

3. Q: How to reshape an array without changing its data?
   A: `array.reshape(new_shape)`

4. Q: How to compute the dot product of two arrays?
   A: `np.dot(array1, array2)` or `array1.dot(array2)`

5. Q: How to generate an array of 5 evenly spaced numbers between 0 and 1?
   A: `np.linspace(0, 1, 5)`

**Q1: How to create an array of zeros with shape 3x3?**

```python
np.zeros((3, 3))
```

**Q2: How to get the sum of all elements in a NumPy array?**

```python
np.sum(arr)
```

**Q3: What function to use for matrix multiplication?**

```python
np.dot(a, b)  # or a @ b
```

**Q4: How do you find the indices of elements greater than a certain value in an array?**

```python
np.where(arr > value)
```
   

## Additional Resources

- [Official NumPy Cheat Sheet](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [DataCamp NumPy Cheat Sheet](https://www.datacamp.com/community/blog/python-numpy-cheat-sheet)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Cheatography NumPy Cheat Sheet](https://cheatography.com/print/numpy-python.pdf)


Remember to check the official documentation for the most up-to-date and detailed information on NumPy functions and usage.


---

This format balances a comprehensive yet concise reference, suitable for GitHub README use and study. You can include diagrams and flowcharts in later updates if necessary.
