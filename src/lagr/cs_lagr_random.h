#ifndef __CS_LAGR_RANDOM_H__
#define __CS_LAGR_RANDOM_H__

/*============================================================================
 * Counter-based random number generation for device particle loops.
 *
 * The legacy cs_random_* API is a stateful lagged-Fibonacci generator:
 * host-only and fundamentally sequential, it cannot be used inside a
 * device parallel_for over particles. This header provides a Philox4x32-10
 * counter-based generator (Random123 family, Salmon et al. 2011):
 *
 *   - stateless: output depends only on (counter, key), so each particle
 *     owns an independent reproducible stream indexed by its id and the
 *     time step — no shared state, no races, no host/device migration;
 *   - host and device compilable (CS_F_HOST_DEVICE, no external calls);
 *   - period 2^128, 2^64 independent streams of length 2^64 — more than
 *     sufficient for 10^9 particles x 10^6 time steps.
 *
 * Statistical quality is NOT bitwise-identical to the host lagged-Fibonacci
 * sequence; validation of ported loops must rely on statistical moments
 * of the Lagrangian fields, not on per-particle trajectory comparison.
 *============================================================================*/

#include <math.h>
#include <stdint.h>

#include "base/cs_defs.h"

/*----------------------------------------------------------------------------
 * Philox4x32-10 round function core.
 * counter: 128-bit as 4x32 (lo->hi), key: 64-bit as 2x32.
 *----------------------------------------------------------------------------*/

inline CS_F_HOST_DEVICE void
_philox4x32_10(uint32_t  ctr[4],
               uint32_t  key[2])
{
  const uint32_t PHILOX_M4x32_0 = 0xD2511F53u;
  const uint32_t PHILOX_M4x32_1 = 0xCD9E8D57u;
  const uint32_t PHILOX_W32_0   = 0x9E3779B9u;
  const uint32_t PHILOX_W32_1   = 0xBB67AE85u;

  for (int round = 0; round < 10; round++) {
    const uint64_t prod0 = (uint64_t)PHILOX_M4x32_0 * (uint64_t)ctr[0];
    const uint64_t prod1 = (uint64_t)PHILOX_M4x32_1 * (uint64_t)ctr[2];

    const uint32_t hi0 = (uint32_t)(prod0 >> 32);
    const uint32_t lo0 = (uint32_t)(prod0);
    const uint32_t hi1 = (uint32_t)(prod1 >> 32);
    const uint32_t lo1 = (uint32_t)(prod1);

    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;

    key[0] += PHILOX_W32_0;
    key[1] += PHILOX_W32_1;
  }
}

/*----------------------------------------------------------------------------
 * Fill out[0..3] with 4 independent uniform uint32 for (counter, key).
 *----------------------------------------------------------------------------*/

inline CS_F_HOST_DEVICE void
cs_lagr_philox4x32(uint32_t  counter_lo,
                   uint32_t  counter_hi,
                   uint32_t  key_lo,
                   uint32_t  key_hi,
                   uint32_t  out[4])
{
  uint32_t ctr[4] = {counter_lo, counter_hi, 0u, 0u};
  uint32_t key[2] = {key_lo, key_hi};
  _philox4x32_10(ctr, key);
  for (int i = 0; i < 4; i++)
    out[i] = ctr[i];
}

/*----------------------------------------------------------------------------
 * Uniform real in [0,1) from a 32-bit random word (53->24 bit precision
 * path kept simple: divide by 2^32; adequate for particle dispersion).
 *----------------------------------------------------------------------------*/

inline CS_F_HOST_DEVICE cs_real_t
cs_lagr_random_uniform(uint32_t  counter_lo,
                       uint32_t  counter_hi,
                       uint32_t  key_lo,
                       uint32_t  key_hi,
                       int       channel)
{
  uint32_t out[4];
  cs_lagr_philox4x32(counter_lo + (uint32_t)(channel / 4),
                     counter_hi, key_lo, key_hi, out);
  return (cs_real_t)out[channel % 4] * (cs_real_t)(1.0/4294967296.0);
}

/*----------------------------------------------------------------------------
 * Standard normal (mean 0, variance 1) via Box-Muller on one Philox block:
 * channels 2k and 2k+1 share the same block but are statistically
 * independent as a pair.
 *----------------------------------------------------------------------------*/

inline CS_F_HOST_DEVICE cs_real_t
cs_lagr_random_normal(uint32_t  counter_lo,
                      uint32_t  counter_hi,
                      uint32_t  key_lo,
                      uint32_t  key_hi,
                      int       channel)
{
  uint32_t out[4];
  cs_lagr_philox4x32(counter_lo + (uint32_t)(channel / 4),
                     counter_hi, key_lo, key_hi, out);
  const int lane = channel % 4;
  const cs_real_t u1 = (cs_real_t)out[lane & ~1]     * (cs_real_t)(1.0/4294967296.0) + (cs_real_t)1e-300;
  const cs_real_t u2 = (cs_real_t)out[(lane & ~1)+1] * (cs_real_t)(1.0/4294967296.0);
  const cs_real_t r  = sqrt(-2.0 * log(u1));
  const cs_real_t th = (cs_real_t)(2.0 * 3.14159265358979323846) * u2;
  return (lane % 2 == 0) ? r * cos(th) : r * sin(th);
}

#endif /* __CS_LAGR_RANDOM_H__ */
