import numpy as np
from dataclasses import dataclass
from typing import Tuple
import struct

@dataclass
class E9S12: # 1 sign bit, 9 exponent bits (bias = 139), 12 significand bits, 5 flag bits
    sign: int  # 0 or 1
    exponent: int  # 0 to 511, biased by 139
    significand: int  # 12 bits, includes implicit bit
    is_zero: bool = False
    is_inf: bool = False
    is_nan: bool = False
    
    EXPONENT_BIAS = 139
    SIGNIFICAND_BITS = 12
    EXPONENT_BITS = 9
    
    # Convert E9S12 to Python float for computation
    def to_float(self) -> float:
        if self.is_zero:
            return -0.0 if self.sign else 0.0
        if self.is_inf:
            return float('-inf') if self.sign else float('inf')
        if self.is_nan:
            return float('nan')
        
        # Compute actual value
        unbiased_exp = self.exponent - self.EXPONENT_BIAS
        significand_value = self.significand / (2 ** (self.SIGNIFICAND_BITS - 1))
        value = significand_value * (2 ** unbiased_exp)
        
        return -value if self.sign else value
    
    def __repr__(self):
        return f"E9S12(sign={self.sign}, exp={self.exponent}, sig=0x{self.significand:03x}, val={self.to_float():.6e})"

# Emulate the paper's floating-point operations
class FloatingPointEmulator:
    # Convert float32 to bit representation   
    @staticmethod
    def float32_to_bits(f) -> int:
        return struct.unpack('>I', struct.pack('>f', f))[0]
    
    # Unpack FP32 into sign, exponent, significand, and flags
    @staticmethod
    def unpack_fp32(value: float) -> Tuple[int, int, int, dict]:
        bits = FloatingPointEmulator.float32_to_bits(value)
        
        sign = (bits >> 31) & 1
        exponent = (bits >> 23) & 0xFF
        fraction = bits & 0x7FFFFF
        
        # Determine if number is zero, infinity, or NaN
        is_zero = (exponent == 0 and fraction == 0)
        is_inf = (exponent == 0xFF and fraction == 0)
        is_nan = (exponent == 0xFF and fraction != 0)
        is_subnormal = (exponent == 0 and fraction != 0)
        
        # Make implicit bit explicit
        if exponent == 0:
            significand = fraction << 1  # Subnormal: no implicit bit
        else:
            significand = (1 << 23) | fraction  # Normal: add implicit bit
        
        flags = {
            'is_zero': is_zero,
            'is_inf': is_inf,
            'is_nan': is_nan,
            'is_subnormal': is_subnormal
        }
        
        return sign, exponent, significand, flags
    
    @staticmethod
    def fp32_to_e9s12_decomposition(value: float) -> Tuple[E9S12, E9S12]:
        sign, exp, sig, flags = FloatingPointEmulator.unpack_fp32(value)
        
        if flags['is_zero']:
            return E9S12(sign, 0, 0, is_zero=True), E9S12(sign, 0, 0, is_zero=True)
        if flags['is_inf']:
            return E9S12(sign, 511, 0, is_inf=True), E9S12(sign, 0, 0, is_zero=True)
        if flags['is_nan']:
            return E9S12(sign, 511, 1, is_nan=True), E9S12(sign, 0, 0, is_zero=True)
        
        # Split 24-bit significand into two 12-bit parts
        # Upper 12 bits -> U_h
        # Lower 12 bits -> U_l
        sig_h = (sig >> 12) & 0xFFF
        sig_l = sig & 0xFFF
        
        # Exponent computation with bias 139
        # U_h_exp = exp + 12 (in biased representation)
        # U_l_exp = exp
        exp_h = exp + 12
        exp_l = exp
        
        # Both parts have the same sign
        u_h = E9S12(sign, exp_h, sig_h, is_zero=(sig_h == 0))
        u_l = E9S12(sign, exp_l, sig_l, is_zero=(sig_l == 0))
        
        return u_h, u_l
        
    @staticmethod
    def demonstrate_decomposition(value: float):
        print("=" * 48)
        print(f"Original value: {value:.15e}")
        
        u_h, u_l = FloatingPointEmulator.fp32_to_e9s12_decomposition(value)
        
        print(f"\nHigh part (U_h):")
        print(f"  {u_h}")
        print(f"\nLow part (U_l):")
        print(f"  {u_l}")
        
        reconstructed = u_h.to_float() + u_l.to_float()
        print(f"\nReconstructed: {reconstructed:.15e}")
        if value != 0 and value != float('inf') and value != -float('inf'):
            print(f"Error: {abs(value - reconstructed):.3e}")
        print("=" * 48)

def main():
    test_values = [np.float32(f) for f in [
            3.14159265358979,
            1.23456789e-5,
            9.87654321e8,
            -2.71828182845905,
            0.0,
            -0.0,
            float('inf'),
            float('-inf'),
            float('nan'),
            1.2345e-40, # Subnormal
            -5.6789e-42, # Subnormal
        ]
    ]

    for val in test_values:
        FloatingPointEmulator.demonstrate_decomposition(val)

if __name__ == "__main__":
    main()