import random
import cmath
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

class TrigonometricNumber(Number):
    def __init__(self, value):
        super().__init__(value)

    def cos(self):
        return TrigonometricNumber(math.cos(self.value))

    def sin(self):
        return TrigonometricNumber(math.sin(self.value))

    def tan(self):
        return TrigonometricNumber(math.tan(self.value))

    def cis(self):
        real = math.cos(self.value)
        imag = math.sin(self.value)
        return ComplexNumber(real, imag)

    def arccos(self):
        return TrigonometricNumber(math.acos(self.value))

    def arcsin(self):
        return TrigonometricNumber(math.asin(self.value))

    def arctan(self):
        return TrigonometricNumber(math.atan(self.value))

    def arccis(self):
        return TrigonometricNumber(math.atan2(self.value.imag, self.value.real))

class NAdicNumber:
    def __init__(self, base, digits):
        self.base = base
        self.digits = digits

    def __add__(self, other):
        if isinstance(other, NAdicNumber) and self.base == other.base:
            max_digits = max(len(self.digits), len(other.digits))
            digits_sum = [((self.digits[i] if i < len(self.digits) else 0) +
                           (other.digits[i] if i < len(other.digits) else 0)) % self.base
                          for i in range(max_digits)]
            return self.__class__(self.base, digits_sum)
        raise TypeError("Unsupported operand type for addition")

    def __sub__(self, other):
        if isinstance(other, NAdicNumber) and self.base == other.base:
            max_digits = max(len(self.digits), len(other.digits))
            digits_diff = [((self.digits[i] if i < len(self.digits) else 0) -
                            (other.digits[i] if i < len(other.digits) else 0)) % self.base
                           for i in range(max_digits)]
            return self.__class__(self.base, digits_diff)
        raise TypeError("Unsupported operand type for subtraction")

    def __mul__(self, other):
        if isinstance(other, NAdicNumber) and self.base == other.base:
            max_digits = len(self.digits) + len(other.digits)
            digits_prod = [0] * max_digits
            for i in range(len(self.digits)):
                for j in range(len(other.digits)):
                    digits_prod[i + j] += (self.digits[i] * other.digits[j]) % self.base
            for i in range(max_digits - 1):
                digits_prod[i + 1] += digits_prod[i] // self.base
                digits_prod[i] %= self.base
            return self.__class__(self.base, digits_prod)
        raise TypeError("Unsupported operand type for multiplication")


class PAdicNumber(NAdicNumber):
    def __init__(self, p, digits):
        super().__init__(p, digits)

    def distance(self, other):
        if isinstance(other, PAdicNumber) and self.base == other.base:
            max_digits = max(len(self.digits), len(other.digits))
            distance = 0
            for i in range(max_digits):
                digit_diff = (self.digits[i] if i < len(self.digits) else 0) - \
                             (other.digits[i] if i < len(other.digits) else 0)
                distance += self.base ** i * abs(digit_diff)
            return distance
        raise TypeError("Unsupported operand type for distance")
        
class DualQuaternion:
    def __init__(self, real_part, dual_part):
        self.real = real_part
        self.dual = dual_part

    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(
                self.real + other.real,
                self.dual + other.dual
            )
        raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(
                self.real - other.real,
                self.dual - other.dual
            )
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            real_part = self.real * other.real - self.dual * other.dual
            dual_part = self.real * other.dual + self.dual * other.real
            return DualQuaternion(real_part, dual_part)
        raise TypeError("Unsupported operand type for *")

class Tessarine:
    def __init__(self, a, b, c, d):
        self.a = a  # Real part
        self.b = b  # Imaginary unit i
        self.c = c  # Imaginary unit j
        self.d = d  # Imaginary unit k

    def __add__(self, other):
        if isinstance(other, Tessarine):
            return Tessarine(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if isinstance(other, Tessarine):
            return Tessarine(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, Tessarine):
            # Perform multiplication according to the tessarine multiplication rules
            real_part = self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d
            i_part = self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
            j_part = self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b
            k_part = self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
            return Tessarine(real_part, i_part, j_part, k_part)
        raise TypeError("Unsupported operand type for *")


class Coquaternion:
    def __init__(self, a, b, c, d):
        self.a = a  # Real part
        self.b = b  # Imaginary unit i
        self.c = c  # Imaginary unit j
        self.d = d  # Imaginary unit k

    def __add__(self, other):
        if isinstance(other, Coquaternion):
            return Coquaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)
        raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if isinstance(other, Coquaternion):
            return Coquaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, Coquaternion):
            # Perform multiplication according to the coquaternion multiplication rules
            real_part = self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d
            i_part = self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
            j_part = self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b
            k_part = self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
            return Coquaternion(real_part, i_part, j_part, k_part)
        raise TypeError("Unsupported operand type for *")


class Biquaternion:
    def __init__(self, a, b, c, d, e, f, g, h, i, j, k):
        self.a = a  # Real part
        self.b = b  # Imaginary unit i
        self.c = c  # Imaginary unit j
        self.d = d  # Imaginary unit k
        self.e = e  # Additional imaginary unit I
        self.f = f  # Additional imaginary unit J
        self.g = g  # Additional imaginary unit K
        self.h = h  # Additional imaginary unit IJ
        self.i = i  # Additional imaginary unit JK
        self.j = j  # Additional imaginary unit KI
        self.k = k  # Additional imaginary unit IJK

    def __add__(self, other):
        if isinstance(other, Biquaternion):
            return Biquaternion(
                self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                self.e + other.e, self.f + other.f, self.g + other.g,
                self.h + other.h, self.i + other.i, self.j + other.j, self.k + other.k
            )
        raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if isinstance(other, Biquaternion):
            return Biquaternion(
                self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d,
                self.e - other.e, self.f - other.f, self.g - other.g,
                self.h - other.h, self.i - other.i, self.j - other.j, self.k - other.k
            )
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, Biquaternion):
            # Perform multiplication according to the biquaternion multiplication rules
            real_part = (
                self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d -
                self.e * other.e - self.f * other.f - self.g * other.g +
                self.h * other.h + self.i * other.i + self.j * other.j + self.k * other.k
            )
            i_part = (
                self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c -
                self.e * other.f - self.f * other.e - self.g * other.h +
                self.h * other.g + self.i * other.j - self.j * other.i - self.k * other.k
            )
            j_part = (
                self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b -
                self.e * other.g + self.f * other.h - self.g * other.e +
                self.h * other.f - self.i * other.k + self.j * other.i + self.k * other.j
            )
            k_part = (
                self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a -
                self.e * other.h + self.f * other.g + self.g * other.f +
                self.h * other.e + self.i * other.k - self.j * other.j + self.k * other.i
            )
            e_part = (
                self.a * other.e + self.b * other.f + self.c * other.g + self.d * other.h +
                self.e * other.a + self.f * other.b + self.g * other.c +
                self.h * other.d + self.i * other.i + self.j * other.j + self.k * other.k
            )
            f_part = (
                self.a * other.f - self.b * other.e + self.c * other.h - self.d * other.g +
                self.e * other.b - self.f * other.a + self.g * other.d -
                self.h * other.c + self.i * other.j - self.j * other.i + self.k * other.k
            )
            g_part = (
                self.a * other.g - self.b * other.h - self.c * other.e + self.d * other.f +
                self.e * other.c - self.f * other.d + self.g * other.a -
                self.h * other.b + self.i * other.k + self.j * other.j - self.k * other.i
            )
            h_part = (
                self.a * other.h + self.b * other.g - self.c * other.f - self.d * other.e +
                self.e * other.d + self.f * other.c - self.g * other.b +
                self.h * other.a + self.i * other.j + self.j * other.i + self.k * other.k
            )
            i_part = (
                self.a * other.i - self.b * other.j + self.c * other.k +
                self.d * other.i - self.e * other.j + self.f * other.k +
                self.g * other.i - self.h * other.j + self.i * other.a -
                self.j * other.b + self.k * other.c
            )
            j_part = (
                self.a * other.j + self.b * other.i - self.c * other.k +
                self.d * other.j + self.e * other.i - self.f * other.k +
                self.g * other.j + self.h * other.i - self.i * other.b +
                self.j * other.a - self.k * other.c
            )
            k_part = (
                self.a * other.k - self.b * other.i + self.c * other.j +
                self.d * other.k - self.e * other.i + self.f * other.j +
                self.g * other.k - self.h * other.i - self.i * other.c +
                self.j * other.b + self.k * other.a
            )
            return Biquaternion(
                real_part, i_part, j_part, k_part,
                e_part, f_part, g_part,
                h_part, i_part, j_part, k_part
            )
        raise TypeError("Unsupported operand type for *")


class Polar:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta

    def __str__(self):
        return f"{self.r}∠{self.theta}"

    def __repr__(self):
        return str(self)

    def to_complex(self):
        real = self.r * math.cos(self.theta)
        imag = self.r * math.sin(self.theta)
        return complex(real, imag)


class Octonion:
    def __init__(self, a, b, c, d, e, f, g, h):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.g = g
        self.h = h

    def __str__(self):
        return f"{self.a} + {self.b}i + {self.c}j + {self.d}k + {self.e}l + {self.f}m + {self.g}n + {self.h}o"

    def __repr__(self):
        return str(self)

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
    valid_chars = "1234567890/*-abcdefghijjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ{()}^+,_"
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
