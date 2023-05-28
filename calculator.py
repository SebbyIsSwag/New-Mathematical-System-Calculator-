import random
import cmath
import math
import statistics
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion






constants = {}

# Define a constant using the 'let' function
def let(name, value):
    constants[name] = value
    
# Define the bottom element
let(Top_Small, float("-inf"))
Top_Small = constants['Top_Small']
# Define the unsigned infinity
let(u_infinity, float("inf")
u_infinity = constants[u_infinity]

# Define 'e' and 'pi' using the 'let' function
let('e', math.e)
let('pi', math.pi)
let('cot', lambda x: 1 / math.tan(x))
let('sec', lambda x: 1 / math.cos(x))



# Access the constants using the defined names
e = constants['e']
pi = constants['pi']
cot = constants['cot']
sec = constants['sec']
def sec(parameter):
    constants['sec']*parameter
def cot(parameter)
    constants['cot']*parameter

class Ordinal:
    def __init__(self, value):
        self.value = value

    def add(self, other):
        return Ordinal(self.value + other.value)

    def multiply(self, other):
        return Ordinal(self.value * other.value)

    def compare(self, other):
        return self.value < other.value

    def prune(self, limit):
        # Prune the ordinal's value to the specified limit
        self.value = min(self.value, limit)

    def to_string(self):
        # Convert the ordinal to its string representation
        return str(self.value)


def omega():
    # Construct and return the ordinal omega
    return Ordinal(math.inf)


def epsilon(n):
    # Construct and return the epsilon ordinal with index n
    if n == 0:
        return Ordinal(0)
    else:
        return Ordinal(1) + epsilon(n-1)


def aleph_null():
    # Construct and return the aleph-null ordinal
    return omega()


def feferman_schutte():
    # Construct and return the Feferman-Schütte ordinal
    return Ordinal(omega().value ** omega().value)


def small_veblen():
    # Construct and return the small Veblen ordinal
    return Ordinal(epsilon(epsilon(0)).value)


def large_veblen():
    # Construct and return the large Veblen ordinal
    return Ordinal(omega().value ** omega().value)


def bachmann_howard():
    # Construct and return the Bachmann-Howard ordinal
    return Ordinal(omega().value ** omega().value)


def buchholz():
    # Construct and return Buchholz's ordinal
    return Ordinal(omega().value ** omega().value)


def takeuti_feferman_buchholz():
    # Construct and return the Takeuti-Feferman-Buchholz ordinal
    return Ordinal(omega().value ** omega().value)


def theories_of_iterated_inductive_definitions():
    # Define and implement operations for theories of iterated inductive definitions
    return Ordinal(omega().value ** omega().value)


def nonrecursive_ordinal():
    # Define and implement operations for nonrecursive ordinals
    return Ordinal(omega().value ** omega().value)




class ZerothRootNumber:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return str(self.x) + "0_r"

    def __add__(self, other):
        if isinstance(other, ZerothRootNumber):
            return ZerothRootNumber(self.x + other.x)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ZerothRootNumber):
            return ZerothRootNumber(self.x - other.x)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, ZerothRootNumber):
            return ZerothRootNumber(self.x * other.x)
        else:
            return NotImplemented

    def __pow__(self, n):
        if n == 0:
            return ZerothRootNumber(1)
        else:
            return self * (self ** (n - 1))

    def __eq__(self, other):
        if isinstance(other, ZerothRootNumber):
            return self.x == other.x
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

def ln(x):
    return math.log(x, math.e)



def log(x, base):
    return math.log(x, base)


def lg(x):
    return math.log10(x)


def dozenal(operation):
    # Perform operation in base-12
    # Replace 10 with 'A' and 11 with 'B'
    operation = operation.replace('10', 'A')
    operation = operation.replace('11', 'B')

    # Convert back to decimal for evaluation
    operation = int(operation, 12)
    return eval(str(operation))


def sexagesimal(operation):
    # Perform operation in base-60
    operation = int(operation, 60)
    return eval(str(operation))


def octal(operation):
    # Perform operation in base-8
    operation = int(operation, 8)
    return eval(str(operation))


def hexadecimal(operation):
    # Perform operation in base-16
    operation = int(operation, 16)
    return eval(str(operation))


def binary(operation):
    # Perform operation in base-2
    operation = int(operation, 2)
    return eval(str(operation))

class Trigintaduonion:
    def __init__(self, components):
        self.components = components
    def __repr__(self):
        return f"Trigintaduonion({self.components})"
    def __add__(self, other):
        if isinstance(other, Trigintaduonion):
            new_components = [c1 + c2 for c1, c2 in zip(self.components, other.components)]
            return Trigintaduonion(new_components))
    def __sub__(self, other):
        if isinstance(other, Trigintaduonion):
            new_components = [c1 - c2 for c1, c2 in zip(self.components, other.components)]
            return Trigintaduonion(new_components)
    def __mul__(self, other):
        if isinstance(other, Trigintaduonion):
            a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = self.components
            A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z = other.components

            result_components = [
                a*A - b*B - c*C - d*D - e*E - f*F - g*G - h*H - i*I - j*J - k*K - l*L - m*M - n*N - o*O - p*P - q*Q - r*R - s*S - t*T - u*U - v*V - w*W - x*X - y*Y - z*Z,
                a*B + b*A + c*D - d*C + e*F - f*E - g*H + h*G + i*J - j*I + k*L - l*K + m*N - n*M - o*P + p*O + q*R - r*Q + s*T - t*S + u*V - v*U + w*X - x*W + y*Z - z*Y,
                a*C - b*D + c*A + d*B + e*G + f*H - g*E - h*F + i*K + j*L - k*I - l*J - m*O + n*P + o*M - p*N + q*Q + r*R - s*O - t*P + u*U + v*V - w*S - x*T - y*Y + z*Z,
                a*D + b*C - c*B + d*A + e*H - f*G + g*F - h*E + i*L + j*K + k*J + l*I - m*P - n*O + o*N + p*M - q*R - r*Q + s*P - t*O + u*V + v*U + w*T - x*S - y*Z - z*Y,
                a*E - b*F - c*G - d*H + e*A + f*B + g*C + h*D + i*N - j*M - k*O - l*P - m*I + n*J + o*K - p*L + q*U - r*V + s*W - t*X + u*S + v*T + w*Y + x*Z - y*Q + z*R,
                a*F + b*E - c*H + d*G - e*B + f*A - g*D + h*C + i*M + j*N - k*P - l*O + m*J - n*I + o*L - p*K - q*V - r*U + s*X + t*W - u*T + v*S - w*Z + x*Y + y*R + z*Q,
                a*G + b*H + c*E - d*F - e*C + f*D + g*A - h*B + i*O + j*P + k*M - l*N - m*K - n*L + o*I + p*J + q*X + r*W - s*U + t*V - u*R + v*Q - w*S - x*Z + y*Y + z*T,
                a*H - b*G + c*F + d*E - e*D - f*C + g*B + h*A + i*P - j*O + k*N + l*M - m*L - n*K - o*J - p*I + q*W + r*X + s*V - t*U + u*Q - v*R + w*Y - x*Z - y*T + z*S,
                a*I - b*J - c*K - d*L - e*M - f*N - g*O - h*P + i*A + j*B + k*C + l*D + m*E + n*F + o*G + p*H + q*Q + r*R + s*S + t*T + u*U + v*V + w*W + x*X + y*Y + z*Z,
                a*J + b*I - c*L + d*K - e*N + f*M - g*P + h*O - i*B + j*A - k*D + l*C - m*F + n*E - o*H + p*G + q*R - r*Q + s*T - t*S + u*V - v*U + w*X - x*W + y*Z - z*Y,
                a*K + b*L + c*I - d*J - e*O - f*P + g*M + h*N - i*C - j*D + k*A + l*B - m*G - n*H + o*E + p*F + q*S + r*T - s*R - t*Q + u*X - v*W + w*U + x*V - y*Y + z*X,
                a*L - b*K + c*J + d*I - e*P + f*O - g*N + h*M - i*D - j*C - k*B + l*A - m*H + n*G - o*F + p*E - q*T + r*S + s*Q - t*R + u*W - v*X + w*V + x*U - y*Z + z*Y,
                a*M - b*N + c*O + d*P - e*I - f*J + g*K + h*L - i*E - j*F - k*G - l*H + m*A + n*B + o*C + p*D + q*U - r*V + s*W - t*X - u*S - v*T - w*Y + x*Z + y*R - z*Q,
                a*N + b*M - c*P + d*O + e*J - f*I - g*L + h*K - i*F + j*E - k*H + l*G + m*B - n*A + o*D - p*C + q*V + r*U - s*X - t*W - u*Q + v*R - w*Z + x*Y + y*S - z*T,
                a*O + b*P - c*N + d*M + e*K + f*L - g*I - h*J - i*G - j*H + k*E + l*F - m*C - n*D + o*A + p*B + q*W + r*X - s*U - t*V - u*R + v*Q + w*Y - x*Z - y*T + z*S,
                a*P - b*O - c*M + d*N + e*L - f*K - g*J + h*I - i*H - j*G - k*F + l*E + m*D - n*C - o*B + p*A - q*X + r*W + s*V - t*U - u*Q + v*R - w*Y + x*Z + y*S - z*T,
                a*Q - b*R - c*S - d*T - e*U - f*V - g*W - h*X - i*Y - j*Z - k*S - l*T - m*U - n*V - o*W - p*X - q*Y - r*Z + s*A + t*B + u*C + v*D + w*E + x*F + y*G + z*H,
                a*R + b*Q - c*T + d*S - e*V + f*U - g*X + h*W - i*Z + j*Y + k*T + l*S - m*X - n*W + o*Q + p*P - q*B + r*A + s*D - t*C + u*F - v*E + w*H - x*G + y*J + z*I,
                a*S + b*T + c*Q - d*R - e*W - f*X + g*U + h*V - i*Y - j*Z + k*U + l*V + m*S + n*T - o*Z - p*Y + q*C + r*D - s*A - t*B + u*G + v*H - w*E - x*F + y*I + z*J,
                a*T - b*S + c*R + d*Q - e*X + f*W - g*V + h*U - i*Z + j*Y - k*V + l*U + m*T - n*S + o*Y + p*Z - q*D - r*C + s*B + t*A - u*H + v*G - w*F - x*E + y*J + z*I,
                a*U - b*V - c*W - d*X - e*S - f*T - g*Q + h*R - i*Z - j*Y + k*V - l*U - m*X - n*W - o*S - p*T + q*F - r*E - s*H + t*G + u*I - v*J + w*C - x*B - y*G + z*F,
                a*V + b*U - c*X + d*W - e*T + f*S - g*R + h*Q - i*Y + j*Z + k*U + l*V + m*W + n*X - o*Z - p*Y - q*G - r*H + s*E + t*F - u*J + v*I - w*B + x*A + y*D - z*C,
                a*W + b*X + c*U - d*V - e*R - f*Q + g*S + h*T - i*X - j*W + k*V + l*U + m*Y + n*Z - o*U - p*V + q*J + r*I - s*F - t*E + u*K + v*L - w*A - x*B + y*H - z*G,
                a*X - b*W + c*V + d*U - e*Q + f*R - g*T + h*S - i*Z + j*Y - k*U + l*V + m*W - n*X + o*Y + p*Z + q*I + r*J - s*E - t*F + u*L - v*K + w*B - x*A + y*G - z*H,
                a*Y - b*Z - c*S - d*T - e*U - f*V - g*W - h*X + i*A + j*B + k*C + l*D + m*H + n*G + o*F + p*E + q*Q + r*R + s*S + t*T + u*X + v*W + w*V + x*U - y*K + z*L,
                a*Z + b*Y - c*T + d*S - e*V + f*U - g*X + h*W + i*B - j*A + k*D - l*C + m*G - n*H + o*E + p*F + q*R - r*Q + s*T - t*S + u*W - v*X + w*V + x*U - y*L + z*K,
                a*A - b*B - c*C - d*D - e*E - f*F - g*G - h*H - i*I - j*J - k*K - l*L - m*M - n*N - o*O - p*P - q*Q - r*R - s*S - t*T - u*U - v*V - w*W - x*X - y*Y - z*Z,
                a*B + b*A + c*D - d*C + e*F - f*E - g*H + h*G + i*J - j*I + k*L - l*K + m*N - n*M - o*P + p*O + q*R - r*Q + s*T - t*S + u*V - v*U + w*X - x*W + y*Z - z*Y,
                a*C - b*D + c*A + d*B + e*G + f*H - g*E - h*F + i*K + j*L - k*I - l*J - m*O + n*P + o*M - p*N + q*Q + r*R - s*O - t*P + u*U + v*V - w*S - x*T - y*Y + z*Z,
                a*D + b*C - c*B + d*A + e*H - f*G + g*F - h*E + i*L + j*K + k*J + l*I - m*P - n*O + o*N + p*M - q*R - r*Q + s*P - t*O + u*V + v*U + w*T - x*S - y*Z - z*Y,
                a*E - b*F - c*G - d*H + e*A + f*B + g*C + h*D + i*N - j*M - k*O - l*P - m*I + n*J + o*K - p*L + q*U - r*V + s*W - t*X + u*S + v*T + w*Y + x*Z - y*Q + z*R,
                a*F + b*E - c*H + d*G - e*B + f*A - g*D + h*C + i*M + j*N - k*P - l*O + m*J - n*I + o*L - p*K - q*V - r*U + s*X + t*W - u*T + v*S - w*Z + x*Y + y*R + z*Q,
                a*G + b*H + c*E - d*F - e*C + f*D + g*A - h*B + i*O + j*P + k*M - l*N - m*K - n*L + o*I + p*J + q*X + r*W - s*U + t*V - u*R + v*Q - w*S - x*Z + y*Y + z*T,
                a*H - b*G + c*F + d*E - e*D - f*C + g*B + h*A + i*P - j*O + k*N + l*M - m*L - n*K - o*J - p*I + q*W + r*X + s*V - t*U + u*Q - v*R + w*Y - x*Z - y*T + z*S,
                a*I - b*J - c*K - d*L - e*M - f*N - g*O - h*P + i*A + j*B + k*C + l*D + m*E + n*F + o*G + p*H + q*Q + r*R + s*S + t*T + u*U + v*V + w*W + x*X + y*Y + z*Z,
                a*J + b*I - c*L + d*K - e*N + f*M - g*P + h*O - i*B + j*A - k*D + l*C - m*F + n*E - o*H + p*G + q*R - r*Q + s*T - t*S + u*V - v*U + w*X - x*W + y*Z - z*Y,
                a*K + b*L + c*I - d*J - e*O - f*P + g*M + h*N - i*C - j*D + k*A + l*B - m*G - n*H + o*E + p*F + q*S + r*T - s*R - t*Q + u*X - v*W + w*U + x*V - y*Y + z*X,
                a*L - b*K + c*J + d*I - e*P + f*O - g*N + h*M - i*D - j*C - k*B + l*A - m*H + n*G - o*F + p*E - q*T + r*S + s*Q - t*R + u*W - v*X + w*V + x*U - y*Z + z*Y,
                a*M - b*N + c*O + d*P - e*I - f*J + g*K + h*L - i*E - j*F - k*G - l*H + m*A + n*B + o*C + p*D + q*U - r*V + s*W - t*X - u*S - v*T - w*Y + x*Z + y*R - z*Q,
                a*N + b*M - c*P + d*O + e*J - f*I - g*L + h*K - i*F + j*E - k*H + l*G + m*B - n*A + o*D - p*C + q*V + r*U - s*X - t*W - u*Q + v*R - w*Z + x*Y + y*S - z*T,
                a*O + b*P - c*N + d*M + e*K + f*L - g*I - h*J - i*G - j*H + k*E + l*F - m*C - n*D + o*A + p*B + q*W + r*X - s*U - t*V - u*R + v*Q + w*Y - x*Z - y*T + z*S,
                a*P - b*O - c*M + d*N + e*L - f*K - g*J + h*I - i*H - j*G - k*F + l*E + m*D - n*C - o*B + p*A - q*X + r*W + s*V - t*U - u*Q + v*R - w*Y + x*Z + y*S - z*T,
                a*Q - b*R - c*S - d*T - e*U - f*V - g*W - h*X - i*Y - j*Z - k*S - l*T - m*U - n*V - o*W - p*X - q*Y - r*Z + s*A + t*B + u*C + v*D + w*E + x*F + y*G + z*H,
                a*R + b*Q - c*T + d*S - e*V + f*U - g*X + h*W - i*Z + j*Y + k*T + l*S - m*X - n*W + o*Q + p*P - q*B + r*A + s*D - t*C + u*F - v*E + w*H - x*G + y*J + z*I,
                a*S + b*T + c*Q - d*R - e*W - f*X + g*U + h*V - i*Y - j*Z + k*U + l*V + m*S + n*T - o*Z - p*Y + q*C + r*D - s*A - t*B + u*G + v*H - w*E - x*F + y*I + z*J,
                a*T - b*S + c*R + d*Q - e*X + f*W - g*V + h*U - i*Z + j*Y - k*V + l*U + m*T - n*S + o*Y + p*Z - q*D - r*C + s*B + t*A - u*H + v*G - w*F - x*E + y*J + z*I,
                a*U - b*V - c*W - d*X - e*S - f*T - g*Q + h*R - i*Z - j*Y + k*V - l*U - m*X - n*W - o*S - p*T + q*F - r*E - s*H + t*G + u*I - v*J + w*C - x*B - y*G + z*F,
                a*V + b*U - c*X + d*W - e*T + f*S - g*R + h*Q - i*Y + j*Z + k*U + l*V + m*W + n*X - o*Z - p*Y - q*G - r*H + s*E + t*F - u*J + v*I - w*B + x*A + y*D - z*C,
                a*W + b*X + c*U - d*V - e*R - f*Q + g*S + h*T - i*X - j*W + k*V + l*U + m*Y + n*Z - o*U - p*V - q*H + r*G + s*F - t*I + u*J + v*B - w*A - x*D + y*C - z*G,
                a*X - b*W + c*V + d*U - e*Q + f*R - g*T + h*S - i*Z + j*Y - k*U + l*V + m*W - n*X + o*Y + p*Z + q*I + r*J - s*E - t*F + u*L - v*K + w*B - x*A + y*G - z*H,
                a*Y - b*Z - c*S - d*T - e*U - f*V - g*W - h*X + i*A + j*B + k*C + l*D + m*H + n*G + o*F + p*E + q*Q + r*R + s*S + t*T + u*X + v*W + w*V + x*U - y*K + z*L,
                a*Z + b*Y - c*T + d*S - e*V + f*U - g*X + h*W + i*B - j*A + k*D - l*C + m*G - n*H + o*E + p*F + q*R - r*Q + s*T - t*S + u*W - v*X + w*V + x*U - y*L + z*K,
                a*A - b*B - c*C - d*D - e*E - f*F - g*G - h*H - i*I - j*J - k*K - l*L - m*M - n*N - o*O - p*P - q*Q - r*R - s*S - t*T - u*U - v*V - w*W - x*X - y*Y - z*Z,
                a*B + b*A + c*D - d*C + e*F - f*E - g*H + h*G + i*J - j*I + k*L - l*K + m*N - n*M - o*P + p*O + q*R - r*Q + s*T - t*S + u*V - v*U + w*X - x*W + y*Z - z*Y,
                a*C - b*D + c*A + d*B + e*G + f*H - g*E - h*F + i*K + j*L - k*I - l*J - m*O + n*P + o*M - p*N + q*Q + r*R - s*O - t*P + u*U + v*V - w*S - x*T - y*Y + z*Z,
                a*D + b*C - c*B + d*A + e*H - f*G + g*F - h*E + i*L + j*K + k*J + l*I - m*P - n*O + o*N + p*M - q*R - r*Q + s*P - t*O + u*V + v*U + w*T - x*S - y*Z - z*Y,
                a*E - b*F - c*G - d*H + e*A + f*B + g*C + h*D + i*N - j*M - k*O - l*P - m*I + n*J + o*K - p*L + q*U - r*V + s*W - t*X + u*S + v*T + w*Y + x






class Sedenion:
    def __init__(self, *components):
        self.components = components
    def __repr__(self):
        return f"Sedenion({self.components})"
    def __add__(self, other):
        if isinstance(self, other):
            new_components = {c + d for c, d in zip(self.components, other.components)}
            return Sedenion(*new_components)
        if isinstance(other, Sedenion):
            new_components = [c1 + c2 for c1, c2 in zip(self.components, other.components)]
            return Sedenion(new_components)
    def __sub__(self, other):
        if isinstance(other, Sedenion):
            new_components = [c1 - c2 for c1, c2 in zip(self.components, other.components)]
            return Sedenion(new_components)
    def __mul__(self, other):
        if isinstance(other, Sedenion):
            if isinstance(type(other), type(self)):
                new_components = [c1 * c2 for c1, c2 in zip(self.components, other.components)]
                return Sedenion(new_components)
            else:
                return NotImplemented



class Number64D:
    def __init__(self, components):
        self.components = components
    def __repr__(self):
        return f"Number64D({self.components})"
    def __add__(self, other):
        if isinstance(other, Number64D):
            new_components = [c1 + c2 for c1, c2 in zip(self.components, other.components)]
            return Number64D(new_components)
        
        if isinstance(other, int):
            new_components = [c1 + other for c1 in self.components]
            return Number64D(new_components)
    
    def __sub__(self, other):
        if isinstance(other, Number64D):
            new_components = [c1 - c2 for c1, c2 in zip(self.components, other.components)]
            return Number64D(new_components)
        
        if isinstance(other, int):
            new_components = [c1 - other for c1 in self.components]
            return Number64D(new_components)
        
    def __mul__(self, other):
        if isinstance(other, Number64D):
            new_components = [c1 * c2 for c1, c2 in zip(self.components, other.components)]
            return Number64D(new_components)
        
        if isinstance(other, int):
            new_components = [c1 * other for c1 in self.components]
            return Number64D(new_components)


class ModularNumber:
    def __init__(self, value, modulus):
        self.value = value % modulus
        self.modulus = modulus
    def __repr__(self):
        return f"{self.value} (mod {self.modulus})"
    def __add__(self, other):
        if isinstance(other, ModularNumber) and self.modulus == other.modulus:
            value = (self.value + other.value) % self.modulus
            return ModularNumber(value, self.modulus)
        return ModularNumber((self.value + other) % self.modulus, self.modulus)
    def __sub__(self, other):
        if isinstance(other, ModularNumber) and self.modulus == other.modulus:
            value = (self.value - other.value) % self.modulus
            return ModularNumber(value, self.modulus)
        return ModularNumber((self.value - other) % self.modulus, self.modulus)
    def __mul__(self, other):
        if isinstance(other, ModularNumber) and self.modulus == other.modulus:
            value = (self.value * other.value) % self.modulus
            return ModularNumber(value, self.modulus)
        return ModularNumber((self.value * other) % self.modulus, self.modulus)
    def mod(self, modulus):
        return ModularNumber(self.value, modulus)


class SplitQuaternion:
    def __init__(self, real, split_imaginary, imaginary_1, imaginary_2):
        self.real = real
        self.split_imaginary = split_imaginary
        self.imaginary_1 = imaginary_1
        self.imaginary_2 = imaginary_2
    def __repr__(self):
        return f"{self.real} + {self.split_imaginary}i + {self.imaginary_1}j + {self.imaginary_2}k"
    def __add__(self, other):
        if isinstance(other, SplitQuaternion):
            real = self.real + other.real
            split_imaginary = self.split_imaginary + other.split_imaginary
            imaginary_1 = self.imaginary_1 + other.imaginary_1
            imaginary_2 = self.imaginary_2 + other.imaginary_2
            return SplitQuaternion(real, split_imaginary, imaginary_1, imaginary_2)
    def __sub__(self, other):
        if isinstance(other, SplitQuaternion):
            real = self.real - other.real
            split_imaginary = self.split_imaginary - other.split_imaginary
            imaginary_1 = self.imaginary_1 - other.imaginary_1
            imaginary_2 = self.imaginary_2 - other.imaginary_2
            return SplitQuaternion(real, split_imaginary, imaginary_1, imaginary_2)
    def __mul__(self, other):
        if isinstance(other, SplitQuaternion):


class PolygonalNumber:
    def __init__(self, n, sides):
        self.n = n
        self.sides = sides

    def value(self):
        return self.n * (self.n - 1) // 2 * (self.sides - 2) + self.n

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def length(self):
        # Calculate the Euclidean distance between the start and end points
        return math.sqrt((self.start.x - self.end.x)**2 + (self.start.y - self.end.y)**2)

class Polygon:
    def __init__(self, vertices):
        self.vertices = vertices

    def perimeter(self):
        perimeter = 0
        num_vertices = len(self.vertices)
        for i in range(num_vertices):
            start = self.vertices[i]
            end = self.vertices[(i + 1) % num_vertices]
            line = Line(start, end)
            perimeter += line.length()
        return perimeter

    def area(self):
        area = 0
        num_vertices = len(self.vertices)
        for i in range(num_vertices):
            current = self.vertices[i]
            next = self.vertices[(i + 1) % num_vertices]
            area += (current.x * next.y - next.x * current.y)
        return abs(area) / 2


class Point5D(Point4D):
    def __init__(self, x, y, z, w, c_5):
        super().__init__(x, y, z, w)
        self.c_5 = c_5

class Point6D(Point5D):
    def __init__(self, x, y, z, w, c_5, c_6):
        super().__init__(x, y, z, w, c_5)
        self.c_6 = c_6

class Point7D(Point6D):
    def __init__(self, x, y, z, w, c_5, c_6, c_7):
        super().__init__(x, y, z, w, c_5, c_6)
        self.c_7 = c_7

class Point8D(Point7D):
    def __init__(self, x, y, z, w, c_5, c_6, c_7, c_8):
        super().__init__(x, y, z, w, c_5, c_6, c_7)
        self.c_8 = c_8

class HyperHyperSphere:
    def __init__(self, center, radius, dimensions):
        self.center = center
        self.radius = radius
        self.dimensions = dimensions

    def volume(self):
        return (math.pi**(self.dimensions/2) / math.gamma(self.dimensions/2 + 1)) * self.radius**self.dimensions

    def surface_area(self):
        return (2 * math.pi**(self.dimensions/2) / math.gamma(self.dimensions/2)) * self.radius**(self.dimensions-1)


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class Point4D(Point3D):
    def __init__(self, x, y, z, w):
        super().__init__(x, y, z)
        self.w = w

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def volume(self):
        return (4/3) * math.pi * self.radius**3

    def surface_area(self):
        return 4 * math.pi * self.radius**2

class Hypersphere(Sphere):
    def __init__(self, center, radius, dimensions):
        super().__init__(center, radius)
        self.dimensions = dimensions

    def volume(self):
        return (math.pi**(self.dimensions/2) / math.gamma(self.dimensions/2 + 1)) * self.radius**self.dimensions

    def surface_area(self):
        return (2 * math.pi**(self.dimensions/2) / math.gamma(self.dimensions/2)) * self.radius**(self.dimensions-1)


class Equation:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def solve(self):
        return self.rhs - self.lhs

class Sum:
    def __init__(self, terms):
        self.terms = terms

    def evaluate(self, x=None, y=None):
        if x is None and y is None:
            return sum(self.terms)
        elif x is not None and y is None:
            return sum(term.evaluate(x) for term in self.terms)
        elif x is not None and y is not None:
            return sum(term.evaluate(x, y) for term in self.terms)
        
class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def rotate(self, angle):
        angle_rad = math.radians(angle)
        new_x = self.x * math.cos(angle_rad) - self.y * math.sin(angle_rad)
        new_y = self.x * math.sin(angle_rad) + self.y * math.cos(angle_rad)
        return Point2D(new_x, new_y)

    def reflect(self):
        return Point2D(-self.x, -self.y)

    def scale(self, factor):
        return Point2D(self.x * factor, self.y * factor)


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"

    def area(self):
        return math.pi * self.radius**2

    def circumference(self):
        return 2 * math.pi * self.radius

    def is_inside(self, point):
        return self.center.distance_to(point) <= self.radius

    def translate(self, dx, dy):
        new_center = Point2D(self.center.x + dx, self.center.y + dy)
        return Circle(new_center, self.radius)


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
