import random
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

# Define the bottom element
Top_Small = float("-inf")

# Define the unsigned infinity
u_infinity = float("inf")

# Define the square root function
def sqrt(x):
    if x < 0:
        return indeterminate(x)
    else:
        return math.sqrt(x)

# Define the cube root function
def cbrt(x):
    return math.pow(x, 1/3)

# Define the nth root function
def nrt(x, n):
    return math.pow(x, 1/n)

# Define the exponentiation function
def power(x, y):
    return math.pow(x, y)

# Define the absolute difference function
def abs_diff(x, y):
    return abs(x - y)

# Define the absolute sum function
def abs_sum(x, y):
    return abs(x + y)

# Define the multiplication function
def multiply(x, y):
    return x * y

# Define the division function
def divide(x, y):
    if y == 0:
        return indeterminate(x)
    else:
        return x / y

# Define the factorial function
def factorial(x):
    return math.factorial(x)

# Define the mean function
def mean(numbers):
    return statistics.mean(numbers)

# Define the random distance function
def random_dis(x, y):
    return random.uniform(x, y)

# Define the define function
def define(n, u):
    return u

# Define the indeterminate function
def indeterminate(x):
    return define(x, random.uniform(Top_Small, u_infinity))

# Function to validate user input
def validate_input(input_str):
    valid_chars = "1234567890/*-abcdefghijjklmnopqrstuvwxyz{()}^+,"
    for char in input_str:
        if char not in valid_chars:
            print("Invalid input! Please enter a valid expression.")
            return None
    return input_str

# Perform a calculation based on the provided expression
def calculate(expression):
    try:
        result = eval(expression)
        return result
    except Exception as e:
        print("Error: Invalid expression!")
        return None

# Input validation for mathematical expressions
def validate_expression(expression):
    valid_chars = "1234567890/*-abcdefghijjklmnopqrstuvwxyz{()}^+,"
    for char in expression:
        if char not in valid_chars:
            print("Error: Invalid character in the expression!")
            return False
    return True

# Validate user input for the mathematical expression
def validate_input(expression):
    try:
        eval(expression)
        return True
    except Exception:
        return False

# Plot a 2D graph
def plot2D(x, y):
    plt.plot(x, y)
    plt.show()

# Plot a 3D graph with 4D stereographic projection
def plot3D(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)
    plt.show()

# Perform matrix operations
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __add__(self, other):
        return np.add(self.matrix, other.matrix)

    def __sub__(self, other):
        return np.subtract(self.matrix, other.matrix)

    def __mul__(self, other):
        return np.dot(self.matrix, other.matrix)

    def __truediv__(self, other):
        return np.divide(self.matrix, other.matrix)

# Perform complex number operations
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)

    def __truediv__(self, other):
        conjugate = ComplexNumber(other.real, -other.imag)
        denominator = other.real**2 + other.imag**2
        real_part = (self.real * other.real + self.imag * other.imag) / denominator
        imag_part = (self.imag * other.real - self.real * other.imag) / denominator
        return ComplexNumber(real_part, imag_part)

# Perform quaternion operations
class QuaternionNumber:
    def __init__(self, scalar, vector):
        self.scalar = scalar
        self.vector = vector

    def __add__(self, other):
        return QuaternionNumber(self.scalar + other.scalar, self.vector + other.vector)

    def __sub__(self, other):
        return QuaternionNumber(self.scalar - other.scalar, self.vector - other.vector)

    def __mul__(self, other):
        scalar_part = self.scalar * other.scalar - np.dot(self.vector, other.vector)
        vector_part = self.scalar * other.vector + other.scalar * self.vector + np.cross(self.vector, other.vector)
        return QuaternionNumber(scalar_part, vector_part)

    def __truediv__(self, other):
        inverse_other = QuaternionNumber(other.scalar, -other.vector)
        result = self * inverse_other
        norm_squared = other.scalar**2 + np.dot(other.vector, other.vector)
        result.scalar /= norm_squared
        result.vector /= norm_squared
        return QuaternionNumber(result.scalar, result.vector)

# Main program loop
while True:
    expression_input = input("Enter the mathematical expression: ")

    if expression_input.lower() == "exit":
        break

    if not validate_input(expression_input):
        print("Invalid input! Please enter a valid expression.")
        continue

    result = calculate(expression_input)
    if result is not None:
        print(f"Result: {result}")

    if isinstance(result, QuaternionNumber):
        x = [result.scalar]
        y = [result.vector[0]]
        z = [result.vector[1]]
        plot3D(x, y, z)
