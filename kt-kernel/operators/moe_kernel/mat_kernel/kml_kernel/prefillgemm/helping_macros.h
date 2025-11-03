#ifndef __HELPING_MACROS_H__
#define __HELPING_MACROS_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if defined(LHS_INT) && defined(LHS_UINT)
#error "Both LHS_INT and LHS_UINT are defined"
#endif

#if defined(RHS_INT) && defined(RHS_UINT)
#error "Both RHS_INT and RHS_UINT are defined"
#endif

#ifdef LHS_INT
#define LHS_TYPE s
#define LHS_INT_TYPE int8_t
#endif
#ifdef LHS_UINT
#define LHS_TYPE u
#define LHS_INT_TYPE uint8_t
#endif
#ifdef RHS_INT
#define RHS_TYPE s
#define RHS_INT_TYPE int8_t
#endif
#ifdef RHS_UINT
#define RHS_TYPE u
#define RHS_INT_TYPE uint8_t
#endif

// mangling macros
#define ADD_M_N_SIZES(name, m_size, n_size) name##_##m_size##x##n_size
#define ADD_M_N_SIZES_MACRO(name, m_size, n_size) ADD_M_N_SIZES(name, m_size, n_size)
#define ADD_TYPES(name, lhs_type, rhs_type) name##_##lhs_type##8##rhs_type##8s32
#define ADD_TYPES_MACRO(name, lhs_type, rhs_type) ADD_TYPES(name, lhs_type, rhs_type)
#define ADD_TYPES_SUFF(name) ADD_TYPES_MACRO(name, LHS_TYPE, RHS_TYPE)
#define ADD_ONE_TYPE_TRANSP(name, type, nt) name##_##type##8_##nt
#define ADD_ONE_TYPE_TRANSP_MACRO(name, type, nt) ADD_ONE_TYPE_TRANSP(name, type, nt)
#define ADD_PACK_A_N_SUFF(name) ADD_ONE_TYPE_TRANSP_MACRO(name, LHS_TYPE, n)
#define ADD_PACK_B_N_SUFF(name) ADD_ONE_TYPE_TRANSP_MACRO(name, RHS_TYPE, n)
#define ADD_PACK_A_T_SUFF(name) ADD_ONE_TYPE_TRANSP_MACRO(name, LHS_TYPE, t)
#define ADD_PACK_B_T_SUFF(name) ADD_ONE_TYPE_TRANSP_MACRO(name, RHS_TYPE, t)
#define ADD_TWO_TYPES_TRANSP(name, lhs_type, rhs_type, a_t, b_t, oc_t) \
  name##_##lhs_type##8##rhs_type##8s32##_##a_t##b_t##_##oc_t
#define ADD_TWO_TYPES_TRANSP_MACRO(name, lhs_type, rhs_type, a_t, b_t, oc_t) \
  ADD_TWO_TYPES_TRANSP(name, lhs_type, rhs_type, a_t, b_t, oc_t)
#define ADD_TRANSP_MACRO(name, a_t, b_t, oc_t) ADD_TWO_TYPES_TRANSP_MACRO(name, LHS_TYPE, RHS_TYPE, a_t, b_t, oc_t)

#ifdef ENABLE_THREADING
#define ADD_THREAD_SUFF(name) name##_thread
#else
#define ADD_THREAD_SUFF(name) name
#endif

#ifdef __cplusplus
}
#endif

#endif