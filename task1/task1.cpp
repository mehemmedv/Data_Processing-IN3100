/*

Task 1
------

The count8 and count64 functions compute the number of entries that
are less than x.

1) Investigate whether GCC can auto-vectorize these functions on
different optimization settings (O3, O2, O1). You can compile this
file as follows:

g++ -march=skylake-avx512 -O3 -std=c++14 -g task1.cpp libjevents.a -o task1

2) Implement count8SIMD and count64SIMD using AVX-512 intrinsics.

3) Check that both versions compute the same result and investigate
which version is most efficient.

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
   for (unsigned i=0; i<inSize; i++)
      if (in[i] < x)
         count++;
   return count;
}
unsigned count64(int64_t* in, unsigned inSize, int64_t x) {
   __asm volatile(""); // make sure this is not optimized away when called multiple times
   unsigned count = 0;
   for (unsigned i=0; i<inSize; i++)
      if (in[i] < x)
         count++;
   return count;
}

unsigned count8SIMD(int8_t* in, unsigned inSize, int8_t x){
   __asm volatile("");
   unsigned count = 0;
   __m512i x_512 = _mm512_set1_epi8 (x);
   
   for(int i = 0; i < inSize; i += 64){
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
   unsigned chunkSize = 512*1024;
   unsigned chunk = (inCount*sizeof(uint8_t)) / chunkSize;
   //for (unsigned i=0; i<chunk; i++)
   
   cout<<(count8(in8, inCount/chunk, 50) ) <<" "<< (   count8SIMD(in8, inCount/chunk, 50))<<endl;
	

   // test
   // for (auto sel : {1, 10, 50, 90, 99}) {
   //   assert(count8(in8, sel, inCount)==count8SIMD(in8, sel, inCount));
   //   assert(count64(in64, sel, inCount)==count64SIMD(in64, sel, inCount));
   //}

   /*PerfEvents e;
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
      
   }*/

   return 0;
}
