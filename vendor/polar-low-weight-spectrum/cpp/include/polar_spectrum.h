#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* H is row-major rows x n, P/S are binary masks of length n.
 * policy: 0 auto, 1 bounded enumeration, 2 syndrome DP.
 * Returns allocated UTF-8 JSON, including an error object on failure.
 * Coefficients are decimal strings. Always release with polar_spectrum_free.
 * Supports n=1,2,...,256. No global mutable state. */
char* polar_spectrum_compute(int n,int rows,const uint8_t* H,int D,
    const uint8_t* P,const uint8_t* S,int policy,uint64_t max_work);
void polar_spectrum_free(char*);
#ifdef __cplusplus
}
#endif
