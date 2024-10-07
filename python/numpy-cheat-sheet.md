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
Here's a comprehensive GitHub README file in markdown format collecting various online cheatsheets for the Python NumPy library:

# NumPy Cheatsheet Collection

This README compiles various online cheatsheets for the NumPy library in Python, providing a quick reference for data scientists, engineers, and Python developers working with numerical computing and data analysis.

## Table of Contents

1. [Interactive NumPy Cheatsheet](#interactive-numpy-cheatsheet)
2. [DataQuest NumPy Cheatsheet](#dataquest-numpy-cheatsheet)
3. [SheCanCode NumPy Cheatsheet](#shecancode-numpy-cheatsheet)
4. [Studyopedia NumPy Cheatsheet](#studyopedia-numpy-cheatsheet)
5. [Python for Data Science Cheatsheet](#python-for-data-science-cheatsheet)
6. [Intellipaat NumPy Cheatsheet](#intellipaat-numpy-cheatsheet)
7. [GeeksforGeeks NumPy Cheatsheet](#geeksforgeeks-numpy-cheatsheet)
8. [DataCamp NumPy Cheatsheet](#datacamp-numpy-cheatsheet)

## Interactive NumPy Cheatsheet

An interactive cheatsheet covering all aspects of the NumPy library, from creating arrays to array manipulation and operations[1].

- **URL**: [https://speedsheet.io/s/numpy](https://speedsheet.io/s/numpy)
- **Features**: 
  - Interactive search functionality
  - Covers array creation, manipulation, and operations
  - Includes examples for each topic

## DataQuest NumPy Cheatsheet

A comprehensive cheatsheet covering key NumPy operations, functions, and techniques[2].

- **URL**: [https://www.dataquest.io/blog/numpy-cheat-sheet/](https://www.dataquest.io/blog/numpy-cheat-sheet/)
- **Topics Covered**:
  - Importing/exporting data
  - Creating arrays
  - Array operations
  - Math functions
  - Array manipulation

## SheCanCode NumPy Cheatsheet

A detailed cheatsheet covering key operations, functions, and techniques for numerical computing in Python using NumPy[3].

- **URL**: [https://shecancode.io/blog/numpy-cheat-sheet-for-python/](https://shecancode.io/blog/numpy-cheat-sheet-for-python/)
- **Sections**:
  - Importing NumPy
  - Creating Arrays
  - Array Operations
  - Array Manipulation
  - Indexing and Slicing

## Studyopedia NumPy Cheatsheet

A guide for working with NumPy, covering basics and advanced topics[4].

- **URL**: [https://studyopedia.com/numpy/numpy-cheat-sheet/](https://studyopedia.com/numpy/numpy-cheat-sheet/)
- **Key Sections**:
  - Installation
  - Creating NumPy Arrays
  - Array Operations
  - Array Shape
  - Searching Arrays

## Python for Data Science Cheatsheet

A comprehensive cheatsheet covering NumPy basics for data science applications[5].

- **URL**: [https://web.itu.edu.tr/iguzel/files/Python_Cheat_Sheets.pdf](https://web.itu.edu.tr/iguzel/files/Python_Cheat_Sheets.pdf)
- **Topics**:
  - Creating Arrays
  - Array Operations
  - Saving & Loading Arrays
  - Array Inspection
  - Sorting Arrays

## Intellipaat NumPy Cheatsheet

A cheatsheet covering array creation, indexing, math operations, and more[6].

- **URL**: [https://intellipaat.com/blog/tutorial/python-tutorial/numpy-cheat-sheet/](https://intellipaat.com/blog/tutorial/python-tutorial/numpy-cheat-sheet/)
- **Sections**:
  - Basic to Advanced Functions
  - Array Creation
  - Array Mathematics
  - Advanced Operations

## GeeksforGeeks NumPy Cheatsheet

A beginner to advanced cheatsheet covering various aspects of NumPy[7].

- **URL**: [https://www.geeksforgeeks.org/numpy-cheat-sheet/](https://www.geeksforgeeks.org/numpy-cheat-sheet/)
- **Key Features**:
  - Array Creation
  - Array Properties Inspection
  - File Handling
  - Array Manipulation
  - Mathematical Operations

## DataCamp NumPy Cheatsheet

A quick reference for NumPy beginners looking to get started with data analysis[8].

- **URL**: [https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python](https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python)
- **Topics Covered**:
  - Array Creation
  - I/O Operations
  - Array Examination
  - Array Mathematics
  - Copying and Sorting Arrays
  - Element Selection
  - Shape Manipulation

These cheatsheets provide a comprehensive overview of NumPy functionality, from basic operations to advanced techniques. They serve as quick references for both beginners and experienced users working with numerical computing and data analysis in Python.

Citations:
[1] https://www.reddit.com/r/Python/comments/repk8q/an_interactive_cheatsheet_for_numpy/

[2] https://www.dataquest.io/blog/numpy-cheat-sheet/

[3] https://shecancode.io/blog/numpy-cheat-sheet-for-python/

[4] https://studyopedia.com/numpy/numpy-cheat-sheet/

[5] https://web.itu.edu.tr/iguzel/files/Python_Cheat_Sheets.pdf

[6] https://intellipaat.com/blog/tutorial/python-tutorial/numpy-cheat-sheet/

[7] https://www.geeksforgeeks.org/numpy-cheat-sheet/

[8] https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python

Remember to check the official documentation for the most up-to-date and detailed information on NumPy functions and usage.


---

This format balances a comprehensive yet concise reference, suitable for GitHub README use and study. You can include diagrams and flowcharts in later updates if necessary.
