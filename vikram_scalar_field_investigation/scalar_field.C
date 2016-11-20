// The libMesh Finite Element Library.
// Copyright (C) 2002-2015 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA



#include "libmesh/getpot.h"

#include "scalarfieldsystem.h"

#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/system.h"
#include "libmesh/equation_systems.h"
#include "libmesh/zero_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"

void ScalarFieldSystem::init_data ()
{
  FEMSystem::init_data();
}



void ScalarFieldSystem::init_context(DiffContext &context)
{
  FEMContext &c = cast_ref<FEMContext&>(context);

  FEBase* elem_fe = NULL;
  c.get_element_fe( 0, elem_fe );

  // Now make sure we have requested all the data
  // we need to build the linear system.
  elem_fe->get_JxW();
  elem_fe->get_dphi();

  // We'll have a more automatic solution to preparing adjoint
  // solutions for time integration, eventually...
  if (c.is_adjoint())
    {
      // A reference to the system context is built with
      const System & sys = c.get_system();

      // Get a pointer to the adjoint solution vector
      NumericVector<Number> &adjoint_solution =
        const_cast<System &>(sys).get_adjoint_solution(0);

      // Add this adjoint solution to the vectors that diff context should localize
      c.add_localized_vector(adjoint_solution0, sys);
    }

  FEMSystem::init_context(context);
}

void ScalarFieldSystem::postprocess()
{
  // Reset the array holding the computed QoIs
  computed_QoI[0] = 0.0;
  computed_QoI[1] = 0.0;
  computed_QoI[2] = 0.0;
  computed_QoI[3] = 0.0;
  computed_QoI[4] = 0.0;
  computed_QoI[5] = 0.0;
  computed_QoI[6] = 0.0;
  computed_QoI[7] = 0.0;
  computed_QoI[8] = 0.0;
  computed_QoI[9] = 0.0;
  computed_QoI[10] = 0.0;
  computed_QoI[11] = 0.0;
  computed_QoI[12] = 0.0;
  computed_QoI[13] = 0.0;
  computed_QoI[14] = 0.0;
  computed_QoI[15] = 0.0;
  computed_QoI[16] = 0.0;
  computed_QoI[17] = 0.0;
  computed_QoI[18] = 0.0;
  computed_QoI[19] = 0.0;
  computed_QoI[20] = 0.0;

  FEMSystem::postprocess();

  //computed_QoI = System::qoi[0];
}
