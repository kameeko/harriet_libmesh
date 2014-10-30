/* The Next Great Finite Element Library. */
/* Copyright (C) 2003  Benjamin S. Kirk */

/* This library is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU Lesser General Public */
/* License as published by the Free Software Foundation; either */
/* version 2.1 of the License, or (at your option) any later version. */

/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU */
/* Lesser General Public License for more details. */

/* You should have received a copy of the GNU Lesser General Public */
/* License along with this library; if not, write to the Free Software */
/* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA */

 // Implementation of high porous media inversion
 // See June 4, 2014 entry
 
// C++ includes
#include <iomanip>
#include <iostream>

// Basic include files
#include "libmesh/equation_systems.h"
#include "libmesh/error_vector.h"
#include "libmesh/getpot.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/uniform_refinement_estimator.h"

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/newton_solver.h"
#include "libmesh/euler_solver.h"
#include "libmesh/steady_solver.h"

// I/O stuff
#include "libmesh/gmv_io.h"
//#include "libmesh/o_string_stream.h"

// Local includes
#include "porous_high_fidelity.h"
#include "initial.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// The main program.
int main (int argc, char** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // This example fails without at least double precision FP
#ifdef LIBMESH_DEFAULT_SINGLE_PRECISION
  libmesh_example_assert(false, "--disable-singleprecision");
#endif

#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_assert(false, "--enable-amr");
#else
      
  // Parse the input file
  GetPot infile("general.in");

  // Read in parameters from the input file
  const Real global_tolerance          = infile("global_tolerance", 0.);
  const unsigned int nelem_target      = infile("n_elements", 400);
  const bool transient                 = infile("transient", true);
  const Real deltat                    = infile("deltat", 0.005);
  const Real timesolver_theta          = infile("timesolver_theta", 1.0);
  unsigned int n_timesteps             = infile("n_timesteps", 20);
  const unsigned int write_interval    = infile("write_interval", 5);
  const unsigned int coarsegridsize    = infile("coarsegridsize", 1);
  const unsigned int coarserefinements = infile("coarserefinements", 0);
  const unsigned int max_adaptivesteps = infile("max_adaptivesteps", 10);  
  
  // Create a mesh.
  Mesh mesh(init.comm());
  
  // And an object to refine it and give it the user specified options
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.coarsen_by_parents() = true;
  mesh_refinement.absolute_global_tolerance() = global_tolerance;
  mesh_refinement.nelem_target() = nelem_target;
  mesh_refinement.refine_fraction() = 0.3;
  mesh_refinement.coarsen_fraction() = 0.3;
  mesh_refinement.coarsen_threshold() = 0.1;

  // Use the MeshTools::Generation mesh generator to create a uniform
  // grid on the square [0,1]^2.  We instruct the mesh generator
  // to build a mesh of 8x8 \p Quad9 elements in 2D. Building these higher-order elements allows
  // us to use higher-order approximation.
  MeshTools::Generation::build_square (mesh,
                                         coarsegridsize,
                                         coarsegridsize,
                                         0., 1.,
                                         0., 1.,
                                         QUAD9);
  
  mesh_refinement.uniformly_refine(coarserefinements);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Declare the system "PorousHFSystem" and its variables.
  PorousHFSystem & system = 
    equation_systems.add_system<PorousHFSystem> ("poroushf_system");

  // Solve this as a time-dependent or steady system
  if (transient)
    {
      EulerSolver *eulersolver =
	new EulerSolver(system);

      eulersolver->theta = timesolver_theta;

      system.time_solver = AutoPtr<TimeSolver>(eulersolver);
    }
  else
    {
      system.time_solver =
        AutoPtr<TimeSolver>(new SteadySolver(system));
      libmesh_assert_equal_to (n_timesteps, 1);
    }

  // Initialize the system  
  equation_systems.init ();  

  libmesh_assert(system.variable_scalar_number(0,0) == 0);
  libmesh_assert(system.variable_scalar_number(1,0) == 1);
  libmesh_assert(system.variable_scalar_number(2,0) == 2);

  libmesh_assert(system.variable_name(0) == "K");
  libmesh_assert(system.variable_name(1) == "p");
  libmesh_assert(system.variable_name(2) == "z");
  
  // Set the initial conditions
  system.project_solution(initial_value, initial_grad,
                                  equation_systems.parameters);

  // Plot the initial conditions
#ifdef LIBMESH_HAVE_GMV
  std::ostringstream file_name_gmv;
  file_name_gmv << "initial.gmv."
                << std::setw(2)
                << std::setfill('0');                

  GMVIO(mesh).write_equation_systems
    (file_name_gmv.str(), equation_systems);    
#endif
  
  // Set the time stepping options
  system.deltat = deltat;

  // And the nonlinear solver options
  NewtonSolver *solver = new NewtonSolver(system);
  system.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver);

  solver->quiet = infile("solver_quiet", true);
  solver->verbose = !solver->quiet;
  solver->max_nonlinear_iterations =
    infile("max_nonlinear_iterations", 15);
  solver->relative_step_tolerance =
    infile("relative_step_tolerance", 1.e-3);
  solver->relative_residual_tolerance =
    infile("relative_residual_tolerance", 0.0);
  solver->absolute_residual_tolerance =
    infile("absolute_residual_tolerance", 0.0);

  solver->require_residual_reduction = infile("require_residual_reduction", false);
      
  // And the linear solver options
  solver->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info();

  // Now we begin the timestep loop to compute the time-accurate
  // solution of the equations.
  for (unsigned int t_step=0; t_step != n_timesteps; ++t_step)
    {
      // A pretty update message
      std::cout << "\n\nSolving time step " << t_step << ", time = "
                << system.time << std::endl;

      // Adaptively solve the timestep
      unsigned int a_step = 0;
      for (; a_step != max_adaptivesteps; ++a_step)
        {
          system.solve();

          system.postprocess();

          ErrorVector error;

          AutoPtr<ErrorEstimator> error_estimator;

          // To solve to a tolerance in this problem we
          // need a better estimator than Kelly
          if (global_tolerance != 0.)
            {
              // We can't adapt to both a tolerance and a mesh
              // size at once
              libmesh_assert_equal_to (nelem_target, 0);

              UniformRefinementEstimator *u =
                new UniformRefinementEstimator;

              // The lid-driven cavity problem isn't in H1, so
              // lets estimate L2 error
              u->error_norm = L2;

              error_estimator.reset(u);
            }
          else
            {
              // If we aren't adapting to a tolerance we need a
              // target mesh size
              libmesh_assert_greater (nelem_target, 0);

              // Kelly is a lousy estimator to use for a problem
              // not in H1 - if we were doing more than a few
              // timesteps we'd need to turn off or limit the
              // maximum level of our adaptivity eventually
              error_estimator.reset(new KellyErrorEstimator);
            }

          // Calculate error based on c_m and c_im <- WHAT ARE c_m AND c_im???? ************
	  std::vector<Real> weights(2,1.0);  // u, v          
          
	  
	  // Keep the same default norm type.
	  std::vector<FEMNormType>
	    norms(1, error_estimator->error_norm.type(0));
	  error_estimator->error_norm = SystemNorm(norms, weights);

          error_estimator->estimate_error(system, error);

          // Print out status at each adaptive step.
          Real global_error = error.l2_norm();
          std::cout << "Adaptive step " << a_step << ": " << std::endl;
          if (global_tolerance != 0.)
            std::cout << "Global_error = " << global_error 
                      << std::endl;
          if (global_tolerance != 0.)
            std::cout << "Worst element error = " << error.maximum()
                      << ", mean = " << error.mean() << std::endl;

          if (global_tolerance != 0.)
            {
              // If we've reached our desired tolerance, we
              // don't need any more adaptive steps
              if (global_error < global_tolerance)
                break;
              mesh_refinement.flag_elements_by_error_tolerance(error);
            }
          else
            {
              // If flag_elements_by_nelem_target returns true, this
              // should be our last adaptive step.
              if (mesh_refinement.flag_elements_by_nelem_target(error))
                {
                  mesh_refinement.refine_and_coarsen_elements();
                  equation_systems.reinit();
                  a_step = max_adaptivesteps;
                  break;
                }
            }

          // Carry out the adaptive mesh refinement/coarsening
          mesh_refinement.refine_and_coarsen_elements();
          equation_systems.reinit();

          std::cout << "Refined mesh to "
                    << mesh.n_active_elem()
                    << " active elements and "
                    << equation_systems.n_active_dofs()
                    << " active dofs." << std::endl;
        }
      // Do one last solve if necessary
      if (a_step == max_adaptivesteps)
        {
          system.solve();

          system.postprocess();
        }

      // Advance to the next timestep in a transient problem
      system.time_solver->advance_timestep();

#ifdef LIBMESH_HAVE_GMV
      std::ostringstream file_name_gmv;
      file_name_gmv << "out.gmv."
		    << std::setw(2)
		    << std::setfill('0')
		    << std::right
		    << t_step;
            
      GMVIO(mesh).write_equation_systems
        (file_name_gmv.str(), equation_systems);    
#endif
    }
#endif // #ifndef LIBMESH_ENABLE_AMR
  
  // All done.  
  return 0;
}
