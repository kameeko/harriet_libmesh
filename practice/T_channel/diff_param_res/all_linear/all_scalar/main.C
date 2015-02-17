//inference with steady coupled stokes-convection-diffusion

// C++ includes
#include <iomanip>

// Basic include files
#include "libmesh/equation_systems.h"
#include "libmesh/error_vector.h"
#include "libmesh/exodusII_io.h"
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/getpot.h"
#include "libmesh/enum_xdr_mode.h" //DEBUG
#include "libmesh/gmv_io.h" //DEBUG

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"
#include "convdiff_inv.h"


int main(int argc, char** argv){

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("fem_system_params.in");
  const Real global_tolerance          = infile("global_tolerance", 0.);
  const unsigned int nelem_target      = infile("n_elements", 400);
  const bool transient                 = infile("transient", true);
  const Real deltat                    = infile("deltat", 0.005);
  unsigned int n_timesteps             = infile("n_timesteps", 1);
  //const unsigned int coarsegridsize    = infile("coarsegridsize", 1);
  const unsigned int coarserefinements = infile("coarserefinements", 0);
  const unsigned int max_adaptivesteps = infile("max_adaptivesteps", 10);
  const unsigned int dim               = 2;
  
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  GetPot infileForMesh("convdiff_inv.in");
  std::string find_mesh_here = infileForMesh("divided_mesh","meep.exo");
  std::cout << find_mesh_here << "\n"; //DEBUG
	mesh.read(find_mesh_here);

  // And an object to refine it
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.coarsen_by_parents() = true;
  mesh_refinement.absolute_global_tolerance() = global_tolerance;
  mesh_refinement.nelem_target() = nelem_target;
  mesh_refinement.refine_fraction() = 0.3;
  mesh_refinement.coarsen_fraction() = 0.3;
  mesh_refinement.coarsen_threshold() = 0.1;

  mesh_refinement.uniformly_refine(coarserefinements);
  
  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  //name system
  ConvDiff_InvSys & system = 
  	equation_systems.add_system<ConvDiff_InvSys>("ConvDiff_InvSys");
  
  //steady-state problem	
 	system.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system));
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
  equation_systems.init ();

  // Set the time stepping options
  system.deltat = deltat; //this is ignored for SteadySolver...right?

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

  // And the linear solver options
  solver->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);
    
  //FOR 1D DEBUG
  //read_initial_parameters();
  //system.project_solution(initial_value, initial_grad, equation_systems.parameters);
  //finish_initialization();
#ifdef LIBMESH_HAVE_GMV
  //GMVIO(equation_systems.get_mesh()).write_equation_systems(std::string("psiHF_readin_1d.gmv"), equation_systems);
#endif

  // Print information about the system to the screen.
  equation_systems.print_info();
  
  // Now we begin the timestep loop to compute the time-accurate
  // solution of the equations...not that this is transient, but eh, why not...
	for (unsigned int t_step=0; t_step != n_timesteps; ++t_step){
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

        // Calculate error
        std::vector<Real> weights(3,1.0); 

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
        
        Number QoI_computed = system.get_QoI_value("computed", 0);
        std::cout<< "Computed QoI is " << std::setprecision(17) << QoI_computed << std::endl;
      }

    // Advance to the next timestep in a transient problem
    system.time_solver->advance_timestep();

#ifdef LIBMESH_HAVE_EXODUS_API
    // Write out this timestep if we're requested to
    if ((t_step+1)%write_interval == 0)
      {
        std::ostringstream file_name;

        // We write the file in the ExodusII format.
        file_name << "out_"
                  << std::setw(3)
                  << std::setfill('0')
                  << std::right
                  << t_step+1
                  << ".e";

        ExodusII_IO(mesh).write_timestep(file_name.str(),
                                         equation_systems,
                                         1, /* This number indicates how many time steps
                                               are being written to the file */
                                         system.time);

      }
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
  }
  
  // All done.
  return 0;
  
} //end main
