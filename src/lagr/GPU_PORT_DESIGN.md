# GPU port of the lagr module — design and staging (2026-11-25)

## Status

- 40 `parallel_for` dispatches done in cell-loop code (coupling, porosity, cs_lagr init, aux_mean) — all compile host-side, hoisting verified by audit.
- `cs_lagr_random.h`: counter-based Philox4x32-10 RNG, host+device, statistically validated. Replaces stateful `cs_random_*` in particle loops.
- **Key fact**: `cs_array` default allocation is `CS_ALLOC_HOST_DEVICE_SHARED` on GPU builds (`default_alloc_mode()` in cs_array.h). The particle buffer `_p_buffer` is therefore ALREADY in shared memory — no manual copy needed. The perf problem is page-fault migration, solved by keeping ALL per-timestep particle loops on device so the buffer migrates once and stays resident.

## What blocks the particle-path kernels (in dependency order)

### 1. Per-particle branchy loops read host-only structures
The injection loop (`cs_lagr_new.cpp:936+`) and SDE integration read:
- `zis` injection-zone structs (host memory, scattered layout)
- `extra_i[phase_id].*` — `->vals[time_id]` double indirection, `->val` field pointers
- `cell_cen`, interpolation limiter arrays
Porting requires hoisting a per-phase "carrier bundle" of raw device pointers BEFORE the kernel — the `cs_lagr_carrier_view_t` infrastructure the user already built is exactly the right vehicle; it must be extended to cover every field the kernel reads.

### 2. RNG stream indexing convention
`cs_random_normal(n*3, vagaus)` batch semantics replaced by per-particle streams:
```
counter_lo = (uint32_t)p_id (or global particle index)
key_lo     = (uint32_t)nt_cur ^ site_salt   (site_salt unique per call site)
channel    = component index (0..2 for xyz)
```
Guarantees: reproducible independent streams, no cross-particle correlation, identical results regardless of GPU thread scheduling. NOT bitwise-identical to host sequence — validation on statistical moments only.

### 3. Dynamic particle count (injection/deletion/compaction)
`resize()` + `memcpy` compaction (cs_lagr_particle.cpp:713) are host. Interim strategy: keep compaction host once per timestep (one migration), all per-substep loops on device. Final: `thrust`-style stream compaction on device.

### 4. Tracking (`cs_lagr_tracking.cpp`, ~4k lines)
Face-crossing events, boundary conditions, adaptive substeps. Port LAST; interim: tracking stays host, positions migrate back once per timestep. This still leaves SDE/integration on device as the dominant cost in most cases.

## Staged plan

| Stage | Content | Validation |
|---|---|---|
| 1 | measure TAYLOR/two-way on GPU machine: confirm lagr share of wall-time | table Timers |
| 2 | extend carrier_view bundle to all kernel inputs (vel, grad_vel, rij, cell_cen, limiters) | compile |
| 3 | port SDE integration inner loop (drag+gravity+Brownian, no crossing) using philox streams | statistical moments vs CPU run |
| 4 | port injection init (branchy but per-particle independent) | idem |
| 5 | device compaction | particle count conservation |
| 6 | tracking (fallback CPU kept permanently) | full regression |

## Conventions for all kernels
- No `[=]` lambda may read file-scope globals, function-statics, or host struct members — hoist to locals.
- No `bft_printf`, no host-only inline helpers on device.
- Scatter loops (particle→cell +=) stay host until atomicAdd infra; reductions stay host.
