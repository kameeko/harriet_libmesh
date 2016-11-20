// Last modified: Oct 9, 2015

// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <cstdlib> // *must* precede <cmath> for proper std:abs() on PGI, Sun Studio CC
#include <cmath>
#include <fstream>

// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/vtk_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/sum_shell_matrix.h"
#include "libmesh/tensor_shell_matrix.h"
#include "libmesh/sparse_shell_matrix.h"
#include "libmesh/mesh_refinement.h"

#include "libmesh/getpot.h"
#include "libmesh/exodusII_io.h"

// To impose Dirichlet boundary conditions
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/zero_function.h"
#include "libmesh/const_function.h"

// This example will solve a linear transient system,
// so we need to include the \p TransientLinearImplicitSystem definition.
#include "libmesh/transient_system.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/vector_value.h"

// The definition of a geometric element
#include "libmesh/elem.h"


// Local function declarations complete, we now move on to the main program

// Bring in everything from the libMesh namespace
using namespace libMesh;

// The main program.
int main (int argc, char** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  std::cout << "Started " << argv[0] << std::endl;

  // Loop over all the seven folders MF00 -> MF06, load psi_LFmesh.xda and superadj.xda from each
  // folder, and compute the error breakdown for each
  for(unsigned int i=0; i <= 6; i++)
  {
    // Create a mesh.
    Mesh mesh (init.comm());

    // And an EquationSystems to run on it
    EquationSystems equation_systems (mesh);

    std::string mesh_file = "/home/vikram/harriet_libmesh/practice/T_channel/diff_param_res/with_reaction/long_channel_stash/qoi3_setup02_r42_basis_blame_all_fine/MF0" + std::to_string(i) + "/psiLF_mesh.xda";

    std::string vars_file = "/home/vikram/harriet_libmesh/practice/T_channel/diff_param_res/with_reaction/long_channel_stash/qoi3_setup02_r42_basis_blame_all_fine/MF0" + std::to_string(i) + "/superadj.xda";

    std::cout << "Reading in mesh" << std::endl;

    // Read in mesh for current iteration
    mesh.read(mesh_file);

    equation_systems.read(vars_file, READ,
    			  EquationSystems::READ_HEADER |
    			  EquationSystems::READ_DATA |
    			  EquationSystems::READ_ADDITIONAL_DATA);

    // Print information about the mesh and systems to the screen.
    mesh.print_info();
    equation_systems.print_info();

    equation_systems.reinit();
  }

  // All done.
  return 0;
}
