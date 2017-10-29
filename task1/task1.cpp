/*

Task 1
------

The count8 and count64 functions compute the number of entries that
are less than x.

1) 
As shown from the results, it can auto-vectorize for count8 and count64

3) It passes the tests(assert)

Send the solution to leis@in.tum.de by Oct 30, 10am.

Include all timeAndProfile output as a comment and the compiler flags
at the end of this file.

 */

#include <iostream>
#include <cassert>
#include <algorithm>
#include <ostream>
#include <vector>
#include <cstring>
#include <sys/mman.h>
#include <cassert>
#include <immintrin.h>

#include "profile.hpp"

using namespace std;

void* malloc_huge(size_t size) {
   void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
   madvise(p, size, MADV_HUGEPAGE);
   memset(p, 0, size);
   return p;
}

unsigned count8(int8_t* in, unsigned inSize, int8_t x) {
   __asm volatile(""); // make sure this is not optimized away when called multiple times
  unsigned count = 0;
   for (unsigned i=0; i<inSize; i++){
      if (in[i] < x)
         count++;
   }
   return count;
}
unsigned count64(int64_t* in, unsigned inSize, int64_t x) {
   __asm volatile(""); // make sure this is not optimized away when called multiple times
   unsigned count = 0;
   for (unsigned i=0; i<inSize; i++){
      if (in[i] < x)
         count++;
   }
   return count;
}

unsigned count8SIMD(int8_t* in, unsigned inSize, int8_t x){
   __asm volatile("");
   unsigned count = 0;
   __m512i x_512 = _mm512_set1_epi8(x);
   
   for(int i = 0; i < inSize; i += 32){
      __m512i cur_nums = _mm512_loadu_si512(in + i);
      __mmask64 mask = _mm512_cmplt_epi8_mask(cur_nums, x_512);
      count += __builtin_popcount(mask);
   }
   return count;
}

unsigned count64SIMD(int64_t* in, unsigned inSize, int64_t x){
   __asm volatile("");
   unsigned count = 0;
   
   __m512i x_512 = _mm512_set1_epi64 (x);
   for(int i = 0; i < inSize; i += 8){
      __m512i cur_nums = _mm512_loadu_si512(in + i);
      __mmask8 mask = _mm512_cmplt_epi64_mask(cur_nums, x_512);
      count += __builtin_popcount(mask);
   }
   return count;
}

int main() {
   int32_t inCount = 1ull << 24;
   unsigned repeat = 200;

   auto in8 = reinterpret_cast<int8_t*>(malloc_huge(inCount*sizeof(int8_t)));
   auto in64 = reinterpret_cast<int64_t*>(malloc_huge(inCount*sizeof(int64_t)));

   for (int32_t i=0; i<inCount; i++) {
      in8[i] = random()%100;
      in64[i] = random()%100;
   }

   // test
   for (auto sel : {1, 10, 50, 90, 99}) {
      assert(count8(in8, inCount, sel)==count8SIMD(in8, inCount, sel));
      assert(count64(in64, inCount, sel)==count64SIMD(in64, inCount, sel));
   }

   PerfEvents e;
   unsigned chunkSize = 512*1024;

   for (auto sel : {1, 10, 50, 90, 99}) {
      e.timeAndProfile("scalar8", inCount, [&]() {
            unsigned chunk = (inCount*sizeof(uint8_t)) / chunkSize;
            for (unsigned i=0; i<chunk; i++)
               assert(count8(in8, inCount/chunk, sel));
         }, repeat, {{"sel", std::to_string(sel)}});
      e.timeAndProfile("scalar64", inCount, [&]() {
            unsigned chunk = (inCount*sizeof(uint64_t)) / chunkSize;
            for (unsigned i=0; i<chunk; i++)
               assert(count64(in64, inCount/chunk, sel));
         }, repeat, {{"sel", std::to_string(sel)}});
      e.timeAndProfile("scalar8SIMD", inCount, [&]() {
            unsigned chunk = (inCount*sizeof(uint8_t)) / chunkSize;
            for (unsigned i=0; i<chunk; i++)
               assert(count8SIMD(in8, inCount/chunk, sel));
         }, repeat, {{"sel", std::to_string(sel)}});
      e.timeAndProfile("scalar64SIMD", inCount, [&]() {
            unsigned chunk = (inCount*sizeof(uint64_t)) / chunkSize;
            for (unsigned i=0; i<chunk; i++)
               assert(count64SIMD(in64, inCount/chunk, sel));
         }, repeat, {{"sel", std::to_string(sel)}});
      
   }

   return 0;
}


/*
g++ -march=skylake-avx512 -O1 -std=c++14 -g task1.cpp libjevents.a -o task1

            name,     sel,  timems, CPUtime,     IPC,     GHz,    BWrd,  cycles, LLCmiss,  L1miss, instruc,  brmiss,  all_rd,  stores,   loads, taskclk,
         scalar8,       1,    8.61,    1.00,    3.27,    4.17,       0,    2.14,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.51,
        scalar64,       1,   15.28,    0.56,    3.26,    4.22,       0,    2.14,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.51,
     scalar8SIMD,       1,    0.39,    1.00,    2.89,    4.16,       0,    0.10,    0.00,    0.02,    0.28,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,       1,    1.84,    1.00,    2.75,    4.16,       0,    0.46,    0.00,    0.13,    1.25,    0.00,    0.00,    0.00,    0.00,    0.11,
         scalar8,      10,    8.63,    1.00,    3.27,    4.16,       0,    2.14,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.51,
        scalar64,      10,    8.59,    1.00,    3.26,    4.19,       0,    2.14,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.51,
     scalar8SIMD,      10,    0.38,    1.00,    2.88,    4.26,       0,    0.10,    0.00,    0.02,    0.28,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      10,    1.79,    1.00,    2.74,    4.27,       0,    0.46,    0.00,    0.13,    1.25,    0.00,    0.00,    0.00,    0.00,    0.11,
         scalar8,      50,    8.42,    1.00,    3.27,    4.26,       0,    2.14,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.50,
        scalar64,      50,    8.27,    1.00,    3.26,    4.35,       0,    2.14,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.49,
     scalar8SIMD,      50,    0.35,    1.00,    2.89,    4.68,       0,    0.10,    0.00,    0.02,    0.28,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      50,    1.63,    1.00,    2.74,    4.68,       0,    0.46,    0.00,    0.13,    1.25,    0.00,    0.00,    0.00,    0.00,    0.10,
         scalar8,      90,    8.04,    1.00,    3.27,    4.47,       0,    2.14,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.48,
        scalar64,      90,    8.05,    1.00,    3.26,    4.47,       0,    2.14,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.48,
     scalar8SIMD,      90,    0.35,    1.00,    2.91,    4.66,       0,    0.10,    0.00,    0.02,    0.28,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      90,    1.64,    1.00,    2.74,    4.66,       0,    0.46,    0.00,    0.13,    1.25,    0.00,    0.00,    0.00,    0.00,    0.10,
         scalar8,      99,    8.03,    1.00,    3.27,    4.47,       0,    2.14,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.48,
        scalar64,      99,    8.05,    1.00,    3.26,    4.47,       0,    2.14,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.48,
     scalar8SIMD,      99,    0.35,    1.00,    2.91,    4.68,       0,    0.10,    0.00,    0.02,    0.28,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      99,    1.63,    1.00,    2.75,    4.68,       0,    0.46,    0.00,    0.13,    1.25,    0.00,    0.00,    0.00,    0.00,    0.10,
*/


/*
g++ -march=skylake-avx512 -O2 -std=c++14 -g task1.cpp libjevents.a -o task1

            name,     sel,  timems, CPUtime,     IPC,     GHz,    BWrd,  cycles, LLCmiss,  L1miss, instruc,  brmiss,  all_rd,  stores,   loads, taskclk,
         scalar8,       1,    7.72,    1.00,    3.65,    4.16,       0,    1.92,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.46,
        scalar64,       1,    7.84,    1.00,    3.60,    4.16,       0,    1.94,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.47,
     scalar8SIMD,       1,    0.44,    1.00,    2.24,    4.24,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,       1,    1.52,    1.00,    2.60,    4.25,       0,    0.39,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.09,
         scalar8,      10,    7.55,    1.00,    3.65,    4.26,       0,    1.92,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.45,
        scalar64,      10,    7.65,    1.00,    3.60,    4.27,       0,    1.94,    0.00,    0.13,    7.00,    0.00,    0.00,    0.00,    0.00,    0.46,
     scalar8SIMD,      10,    0.44,    1.00,    2.24,    4.26,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,      10,    1.52,    1.00,    2.59,    4.27,       0,    0.39,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.09,
         scalar8,      50,    7.42,    1.00,    3.65,    4.33,       0,    1.92,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.44,
        scalar64,      50,    7.30,    1.00,    3.60,    4.47,       0,    1.95,    0.00,    0.12,    7.00,    0.00,    0.00,    0.00,    0.00,    0.43,
     scalar8SIMD,      50,    0.40,    1.00,    2.25,    4.68,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      50,    1.38,    1.00,    2.60,    4.66,       0,    0.39,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.08,
         scalar8,      90,    7.19,    1.00,    3.65,    4.47,       0,    1.92,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.43,
        scalar64,      90,    7.30,    1.00,    3.60,    4.47,       0,    1.95,    0.00,    0.12,    7.00,    0.00,    0.00,    0.00,    0.00,    0.44,
     scalar8SIMD,      90,    0.40,    1.00,    2.25,    4.68,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      90,    1.38,    1.00,    2.59,    4.68,       0,    0.39,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.08,
         scalar8,      99,    7.19,    1.00,    3.65,    4.47,       0,    1.92,    0.00,    0.02,    7.00,    0.00,    0.00,    0.00,    0.00,    0.43,
        scalar64,      99,    7.30,    1.00,    3.60,    4.47,       0,    1.94,    0.00,    0.12,    7.00,    0.00,    0.00,    0.00,    0.00,    0.44,
     scalar8SIMD,      99,    0.40,    1.00,    2.25,    4.68,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      99,    1.38,    1.00,    2.60,    4.68,       0,    0.38,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.08,
*/

/*
g++ -march=skylake-avx512 -O3 -std=c++14 -g task1.cpp libjevents.a -o task1

            name,     sel,  timems, CPUtime,     IPC,     GHz,    BWrd,  cycles, LLCmiss,  L1miss, instruc,  brmiss,  all_rd,  stores,   loads, taskclk,
         scalar8,       1,    0.67,    1.00,    1.79,    4.16,       0,    0.17,    0.00,    0.02,    0.30,    0.00,    0.00,    0.00,    0.00,    0.04,
        scalar64,       1,    0.87,    1.00,    3.19,    4.16,       0,    0.22,    0.00,    0.13,    0.69,    0.00,    0.00,    0.00,    0.00,    0.05,
     scalar8SIMD,       1,    0.45,    1.00,    2.24,    4.16,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,       1,    1.55,    1.00,    2.60,    4.16,       0,    0.38,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.09,
         scalar8,      10,    0.67,    1.00,    1.80,    4.16,       0,    0.17,    0.00,    0.02,    0.30,    0.00,    0.00,    0.00,    0.00,    0.04,
        scalar64,      10,    0.86,    1.00,    3.22,    4.16,       0,    0.21,    0.00,    0.13,    0.69,    0.00,    0.00,    0.00,    0.00,    0.05,
     scalar8SIMD,      10,    0.45,    1.00,    2.25,    4.16,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,      10,    1.55,    1.00,    2.60,    4.16,       0,    0.38,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.09,
         scalar8,      50,    0.67,    1.00,    1.79,    4.16,       0,    0.17,    0.00,    0.02,    0.30,    0.00,    0.00,    0.00,    0.00,    0.04,
        scalar64,      50,    0.85,    1.00,    3.20,    4.22,       0,    0.22,    0.00,    0.12,    0.69,    0.00,    0.00,    0.00,    0.00,    0.05,
     scalar8SIMD,      50,    0.44,    1.00,    2.24,    4.24,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,      50,    1.51,    1.00,    2.60,    4.27,       0,    0.38,    0.00,    0.13,    1.00,    0.00,    0.00,    0.00,    0.00,    0.09,
         scalar8,      90,    0.65,    1.00,    1.79,    4.27,       0,    0.17,    0.00,    0.02,    0.30,    0.00,    0.00,    0.00,    0.00,    0.04,
        scalar64,      90,    0.85,    1.00,    3.17,    4.27,       0,    0.22,    0.00,    0.12,    0.69,    0.00,    0.00,    0.00,    0.00,    0.05,
     scalar8SIMD,      90,    0.44,    1.00,    2.24,    4.26,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.03,
    scalar64SIMD,      90,    1.40,    1.00,    2.60,    4.60,       0,    0.39,    0.00,    0.12,    1.00,    0.00,    0.00,    0.00,    0.00,    0.08,
         scalar8,      99,    0.60,    1.00,    1.80,    4.65,       0,    0.17,    0.00,    0.02,    0.30,    0.00,    0.00,    0.00,    0.00,    0.04,
        scalar64,      99,    0.77,    1.00,    3.19,    4.68,       0,    0.22,    0.00,    0.12,    0.69,    0.00,    0.00,    0.00,    0.00,    0.05,
     scalar8SIMD,      99,    0.40,    1.00,    2.25,    4.67,       0,    0.11,    0.00,    0.02,    0.25,    0.00,    0.00,    0.00,    0.00,    0.02,
    scalar64SIMD,      99,    1.38,    1.00,    2.60,    4.68,       0,    0.39,    0.00,    0.12,    1.00,    0.00,    0.00,    0.00,    0.00,    0.08,
*/
