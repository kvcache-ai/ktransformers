#ifndef BETA_MACROS_H
#define BETA_MACROS_H

#if defined(OC_FIX)
#define OC_TYPE f
#define OC_IDX(mi, nn) 0
#elif defined(OC_COL)
#define OC_TYPE c
#define OC_IDX(mi, ni) mi
#else
#define OC_TYPE r
#define OC_IDX(mi, ni) ni
#endif

#if defined(BETA_OPT)
#define BETA_SUFF(name) name##_opt
#define LDC(m, ldc) ldc
#define DTYPE int32_t
#else
#define BETA_SUFF(name) name
#define LDC(m, ldc) m
#define DTYPE float
#endif

#endif