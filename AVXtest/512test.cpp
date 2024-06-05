/*
# Copyright H. Watanabe 2017
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
*/
//------------------------------------------------------------------------
#include <immintrin.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <random>
#include <sstream>
//------------------------------------------------------------------------
static std::mt19937 mt(1);
std::uniform_int_distribution<int> ud(0, 1);
//------------------------------------------------------------------------
__attribute__((noinline))
__m128i ternlog(__m128i x1, __m128i x2, __m128i x3) {
  __asm__ (
    "vpternlogd $22,%zmm2,%zmm1,%zmm0\n\t"
  );
}
//------------------------------------------------------------------------
__int128 rand_bit(void) {
  __int128 v = 0;
  for (int i = 0; i < 128; i++) {
    v = v << 1;
    v |= ud(mt);
  }
  return v;
}
//------------------------------------------------------------------------
int
str2bit(std::string str) {
  int r = 0;
  for (int i = 0; i < 8; i++) {
    r = r + ((str[i] - '0') << i);
  }
  return r;
}
//------------------------------------------------------------------------
std::string
i2b(unsigned int a) {
  std::stringstream ss;
  for (int i = 0; i < 32; i++) {
    ss << (a & 1) ? "1" : "0";
    a = a >> 1;
  }
  return ss.str();
}
//------------------------------------------------------------------------
std::string
i2b(__int128 a) {
  std::stringstream ss;
  for (int i = 0; i < 128; i++) {
    ss << ((a & 1) ? "1" : "0");
    a = a >> 1;
  }
  return ss.str();
}
//------------------------------------------------------------------------
std::string
i2b(__m128i x) {
  unsigned int *a = (unsigned int *)(&x);
  std::string s;
  for (int i = 0; i < 4; i++) {
    s = s + i2b(a[i]);
  }
  return s;
}
//------------------------------------------------------------------------
std::string
i2b(__m512i x) {
  unsigned int *a = (unsigned int *)(&x);
  std::string s;
  for (int i = 0; i < 4; i++) {
    s = s + i2b(a[i]);
  }
  return s;
}
//------------------------------------------------------------------------
__int128
ternlog512(__int128 &t1, __int128 &t2, __int128 &t3) {
  __m512i z1 = _mm512_load_si512((__m512i*)(&t1));
  __m512i z2 = _mm512_load_si512((__m512i*)(&t2));
  __m512i z3 = _mm512_load_si512((__m512i*)(&t3));
  __m512i z4 = _mm512_ternarylogic_epi32(z1, z2, z3, 22);
  __attribute__((aligned(64))) __int128 t[4];
  _mm512_store_si512((__m512i*)(t), z4);
  return t[0];
}
//------------------------------------------------------------------------
int
main(void) {
  __attribute__((aligned(64))) __int128 t1 = rand_bit();
  __attribute__((aligned(64))) __int128 t2 = rand_bit();
  __attribute__((aligned(64))) __int128 t3 = rand_bit();
  std::cout << "Input" << std::endl;
  std::cout << i2b(t1) << std::endl;
  std::cout << i2b(t2) << std::endl;
  std::cout << i2b(t3) << std::endl;
  std::cout << "Output" << std::endl;
  // Scalar
  __int128 t4 = (t1 & t2 & t3) ^ (t1 ^ t2 ^ t3);
  std::cout << i2b(t4) << std::endl;
  __m128i x1 = _mm_load_si128((__m128i*)(&t1));
  __m128i x2 = _mm_load_si128((__m128i*)(&t2));
  __m128i x3 = _mm_load_si128((__m128i*)(&t3));
  //__m128i x4 = _mm_ternarylogic_epi32(x1,x2,x3,22); // AVXVL

  // Inline Assembler
  __m128i x4 = ternlog(x1, x2, x3);
  std::cout << "aaa" << std::endl;
  std::cout << i2b(x4) << std::endl;

  // Using zmm
  
  __m512i z1 = _mm512_load_si512((__m512i*)(&t1));
  __m512i z2 = _mm512_load_si512((__m512i*)(&t2));
  __m512i z3 = _mm512_load_si512((__m512i*)(&t3));
  __m512i z4 = _mm512_ternarylogic_epi32(z1, z2, z3, 22);
  __attribute__((aligned(64))) __int128 t5[4];
  _mm512_store_si512((__m512i*)(t5), z4);
  std::cout << i2b(t5[0]) << std::endl;
  //Optimized?
  std::cout << i2b(ternlog512(t1, t2, t3)) << std::endl;
}
//------------------------------------------------------------------------
/* Expected Result
$ icpc -xMIC-AVX512 test.cpp
$ ./a.out
Input
10011010010110111010001011001000100100000100011111000111000101011010110000101111110011111010010011100100110011101000000010001110
01110101001000000011010111000101100001111100011100110110011001111000110001100101011011001001111101111001101001100011110111010110
10001111111001000001000100100001110110010110011110001010111100001101011001000001010000110110101010000001011100111001100101110000
Output
01100000100111111000011000101100010011101010000001111001100000100111001000001010101000000101000100011100000110010010010000101000
01100000100111111000011000101100010011101010000001111001100000100111001000001010101000000101000100011100000110010010010000101000
01100000100111111000011000101100010011101010000001111001100000100111001000001010101000000101000100011100000110010010010000101000
01100000100111111000011000101100010011101010000001111001100000100111001000001010101000000101000100011100000110010010010000101000
*/
//------------------------------------------------------------------------