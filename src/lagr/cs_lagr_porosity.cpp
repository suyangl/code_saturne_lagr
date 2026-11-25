/*============================================================================
 * Handling of porosity due to particle deposition.
 *============================================================================*/

/*
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2026 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/*----------------------------------------------------------------------------*/

/*============================================================================
 * Functions dealing with particle tracking
 *============================================================================*/

#include "base/cs_defs.h"

/*----------------------------------------------------------------------------
 * Standard C library headers
 *----------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/*----------------------------------------------------------------------------
 *  Local headers
 *----------------------------------------------------------------------------*/

#include "bft/bft_printf.h"
#include "bft/bft_error.h"

#include "base/cs_base.h"
#include "base/cs_dispatch.h"
#include "base/cs_halo.h"
#include "base/cs_math.h"
#include "base/cs_mem.h"
#include "mesh/cs_mesh.h"
#include "mesh/cs_mesh_quantities.h"
#include "base/cs_parall.h"
#include "base/cs_porous_model.h"

#include "lagr/cs_lagr.h"
#include "lagr/cs_lagr_particle.h"

/*----------------------------------------------------------------------------
 *  Header for the current file
 *----------------------------------------------------------------------------*/

#include "lagr/cs_lagr_porosity.h"

/*----------------------------------------------------------------------------*/

/*! \cond DOXYGEN_SHOULD_SKIP_THIS */

/*=============================================================================
 * Local Macro definitions
 *============================================================================*/

/*=============================================================================
 * Local Enumeration definitions
 *============================================================================*/

/*============================================================================
 * Local structure definitions
 *============================================================================*/

/*============================================================================
 * Static global variables
 *============================================================================*/

/*=============================================================================
 * Private function definitions
 *============================================================================*/

/*! (DOXYGEN_SHOULD_SKIP_THIS) \endcond */

/*============================================================================
 * Public function definitions
 *============================================================================*/

/*----------------------------------------------------------------------------*/
/*!
 * \brief Compute interior face porosity for Lagrangian internal deposition.
 */
/*----------------------------------------------------------------------------*/

void
cs_lagr_porosity(void)
{
  const cs_mesh_t *mesh = cs_glob_mesh;
  const cs_mesh_quantities_t  *fvq = cs_glob_mesh_quantities;
  const cs_mesh_quantities_t  *mq_g = cs_glob_mesh_quantities_g;
  const cs_lagr_model_t *lagr_model = cs_glob_lagr_model;

  cs_lagr_particle_set_t& p_set = cs_lagr_get_particle_set_ref();

  cs_lagr_internal_condition_t *internal_conditions
    = cs_glob_lagr_internal_conditions;

  if (internal_conditions == nullptr)
    return;

  const cs_lnum_t n_cells_ext = mesh->n_cells_with_ghosts;
  const cs_lnum_t n_i_faces = mesh->n_i_faces;

  /* Separate host/device buffers: the particle scatter below writes the
     host buffer, a single explicit copy replaces unified-memory page
     fault migrations. */
  cs_real_t *covered_surface = nullptr;
  CS_MALLOC_HD(covered_surface, mesh->n_cells_with_ghosts, cs_real_t,
               CS_ALLOC_HOST_DEVICE);

  cs_dispatch_context ctx;

  /* Initialization on host (scatter below accumulates on host) */
  cs_array_real_fill_zero(n_cells_ext, covered_surface);

  /* Local copies for the device lambda (host struct members) */
  cs_real_t *i_face_normal = fvq->i_face_normal;
  const cs_real_t *i_face_normal_g = mq_g->i_face_normal;
  const int *i_face_zone_id = internal_conditions->i_face_zone_id;

  cs_real_t *d_i_face_normal
    = (cs_real_t *)cs_get_device_ptr(i_face_normal);
  const cs_real_t *d_i_face_normal_g
    = (const cs_real_t *)cs_get_device_ptr_const(i_face_normal_g);

  ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
    /* Internal face flagged as internal deposition */
    if (i_face_zone_id[face_id] >= 0) {
      for (cs_lnum_t j = 0; j < 3; j++)
        d_i_face_normal[3*face_id+j] = d_i_face_normal_g[3*face_id+j];
    }
  });
  ctx.wait();

  if (lagr_model->deposition == 1) {

    for (cs_lnum_t p_id = 0; p_id < p_set.n_particles; p_id++) {

      if (p_set.flag(p_id, CS_LAGR_PART_IMPOSED_MOTION)) {

        cs_lnum_t cell_id
          = p_set.attr_lnum(p_id, CS_LAGR_CELL_ID);

        cs_real_t diam2
          = cs_math_pow2(p_set.attr_real(p_id, CS_LAGR_DIAMETER));

        covered_surface[cell_id] += cs_math_pi * 0.25 * diam2
          * p_set.attr_real(p_id, CS_LAGR_FOULING_INDEX)
          * p_set.attr_real(p_id, CS_LAGR_STAT_WEIGHT);

      }
    }
  }

  /* Synchronization */
  if (mesh->halo != nullptr)
    cs_halo_sync_var(mesh->halo, CS_HALO_STANDARD, covered_surface);

  /* Upload scattered host buffer to device (single copy) */
  cs_sync_h2d(covered_surface);

  /* Compute fluid section and clip it to 0 if negative */
  const cs_lnum_2_t *i_face_cells = (const cs_lnum_2_t *)(mesh->i_face_cells);
  cs_real_t *i_face_surf = fvq->i_face_surf;

  const cs_real_t *d_covered_surface
    = (const cs_real_t *)cs_get_device_ptr_const(covered_surface);
  cs_real_t *d_i_face_surf
    = (cs_real_t *)cs_get_device_ptr(i_face_surf);

  ctx.parallel_for(n_i_faces, [=] CS_F_HOST_DEVICE (cs_lnum_t face_id) {
    /* Internal face flagged as internal deposition */
    if (i_face_zone_id[face_id] >= 0) {
      cs_lnum_t cell_id1 = i_face_cells[face_id][0];
      cs_lnum_t cell_id2 = i_face_cells[face_id][1];
      /* Remove from the particle area from fluid section */
      for (cs_lnum_t id = 0; id < 3; id++)
        d_i_face_normal[3*face_id + id]
          -= (d_covered_surface[cell_id1] + d_covered_surface[cell_id2])
             * d_i_face_normal_g[3*face_id + id]
             / d_i_face_surf[face_id];

      /* If S_fluid . S is negative, that means we removed too much surface
       * from fluid surface */
      cs_real_t temp
        = cs_math_3_dot_product(d_i_face_normal + 3*face_id,
                                d_i_face_normal_g + 3*face_id);
      if (temp <= 0.) {
        for (cs_lnum_t j = 0; j < 3; j++)
          d_i_face_normal[3*face_id+j] = 0.;
      }
      d_i_face_surf[face_id]
        = cs_math_3_norm(d_i_face_normal + 3*face_id);
    }
  });
  ctx.wait();

  CS_FREE(covered_surface);
}

/*----------------------------------------------------------------------------*/
