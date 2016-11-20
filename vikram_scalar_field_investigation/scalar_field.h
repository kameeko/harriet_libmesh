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



#include "libmesh/enum_fe_family.h"
#include "libmesh/fem_system.h"
#include "libmesh/parameter_vector.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class ScalarFieldSystem : public FEMSystem
{
public:
  // Constructor
  ScalarFieldSystem(EquationSystems& es,
             const std::string& name_in,
             const unsigned int number_in)
    : FEMSystem(es, name_in, number_in) { qoi.resize(20); }

  Number &get_QoI_value(unsigned int QoI_index)
  {
    return computed_QoI[QoI_index];
  }

protected:
  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context (DiffContext &context);

  // Postprocess functions
  virtual void element_postprocess (DiffContext &context);

  // Variables to hold the computed QoIs
  Number computed_QoI[20];
};
