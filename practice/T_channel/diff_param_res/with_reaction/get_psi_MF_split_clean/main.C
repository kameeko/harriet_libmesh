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
#include "libmesh/direct_solution_transfer.h"

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"
#include "convdiff_mprime.h"
#include "convdiff_primary.h"
#include "convdiff_aux.h"


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
  //const unsigned int dim               = 2;
  
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  Mesh mesh2(init.comm());
  GetPot infileForMesh("convdiff_mprime.in");
  std::string find_mesh_here = infileForMesh("divided_mesh","meep.exo");
	mesh.read(find_mesh_here);
	mesh2.read(find_mesh_here);
	//mesh.read("psiHF_mesh_1Dfused.xda");

	bool doContinuation = infileForMesh("do_continuation",false);

  // And an object to refine it
  /*MeshRefinement mesh_refinement(mesh);
  mesh_refinement.coarsen_by_parents() = true;
  mesh_refinement.absolute_global_tolerance() = global_tolerance;
  mesh_refinement.nelem_target() = nelem_target;
  mesh_refinement.refine_fraction() = 0.3;
  mesh_refinement.coarsen_fraction() = 0.3;
  mesh_refinement.coarsen_threshold() = 0.1;

  mesh_refinement.uniformly_refine(coarserefinements);*/
  
  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  EquationSystems equation_systems_mix(mesh2);
  
  //name system
  ConvDiff_PrimarySys & system_primary = 
  	equation_systems.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys");
	ConvDiff_AuxSys & system_aux = 
  	equation_systems.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys");
  ConvDiff_MprimeSys & system_mix = 
		equation_systems_mix.add_system<ConvDiff_MprimeSys>("ConvDiff_MprimeSys");

  //steady-state problem	
 	system_primary.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_primary));
  system_aux.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_aux));
  system_mix.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_mix));
  libmesh_assert_equal_to (n_timesteps, 1);

	if(doContinuation){
		equation_systems.read("psiLF_split.xda", READ,
				EquationSystems::READ_HEADER |
				EquationSystems::READ_DATA |
				EquationSystems::READ_ADDITIONAL_DATA);
		equation_systems.print_info();
  }
  else{
		// Initialize the system
		equation_systems.init ();
	}

  // Set the time stepping options
  system_primary.deltat = deltat; system_aux.deltat = deltat;//this is ignored for SteadySolver...right?

  // And the nonlinear solver options
  NewtonSolver *solver_primary = new NewtonSolver(system_primary); 
  system_primary.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_primary); 
  solver_primary->quiet = infile("solver_quiet", true);
  solver_primary->verbose = !solver_primary->quiet;
  solver_primary->max_nonlinear_iterations =
    infile("max_nonlinear_iterations", 15);
  solver_primary->relative_step_tolerance =
    infile("relative_step_tolerance", 1.e-3);
  solver_primary->relative_residual_tolerance =
    infile("relative_residual_tolerance", 0.0);
  solver_primary->absolute_residual_tolerance =
    infile("absolute_residual_tolerance", 0.0);
  NewtonSolver *solver_aux = new NewtonSolver(system_aux); 
  system_aux.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_aux); 
  solver_aux->quiet = infile("solver_quiet", true);
  solver_aux->verbose = !solver_aux->quiet;
  solver_aux->max_nonlinear_iterations =
    infile("max_nonlinear_iterations", 15);
  solver_aux->relative_step_tolerance =
    infile("relative_step_tolerance", 1.e-3);
  solver_aux->relative_residual_tolerance =
    infile("relative_residual_tolerance", 0.0);
  solver_aux->absolute_residual_tolerance =
    infile("absolute_residual_tolerance", 0.0);
  //solver_aux->require_residual_reduction = false; //DEBUG

  // And the linear solver options
  solver_primary->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver_primary->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);
 	solver_aux->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver_aux->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info();


  // Now we begin the timestep loop to compute the time-accurate
  // solution of the equations...not that this is transient, but eh, why not...
	for (unsigned int t_step=0; t_step != n_timesteps; ++t_step){
    // A pretty update message
    std::cout << "\n\nSolving time step " << t_step << ", time = "
              << system_primary.time << std::endl;

    // Adaptively solve the timestep
    unsigned int a_step = 0;
    /*for (; a_step != max_adaptivesteps; ++a_step)
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
        std::vector<Real> weights(9,1.0);  // based on u, v, p, c, their adjoints, and source parameter

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
      }*/
    // Do one last solve if necessary
    if (a_step == max_adaptivesteps)
      {
        system_primary.solve();
				std::cout << "\n\n Residual L2 norm (primary): " 
					<< system_primary.calculate_norm(*system_primary.rhs, L2) << std::endl;
				std::cout << "\n\n Solution L2 norm (primary): " 
					<< system_primary.calculate_norm(*system_primary.solution, L2) << std::endl << std::endl;
				system_aux.solve();
				std::cout << "\n\n Residual L2 norm (auxiliary): " << system_aux.calculate_norm(*system_aux.rhs, L2) << "\n";
				
				equation_systems_mix.init();
  			DirectSolutionTransfer sol_transfer(init.comm()); 
  			sol_transfer.transfer(system_primary.variable(system_primary.variable_number("c")),
  				system_mix.variable(system_mix.variable_number("c")));
				sol_transfer.transfer(system_primary.variable(system_primary.variable_number("zc")),
  				system_mix.variable(system_mix.variable_number("zc")));
				sol_transfer.transfer(system_primary.variable(system_primary.variable_number("fc")),
  				system_mix.variable(system_mix.variable_number("fc")));
				sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_c")),
  				system_mix.variable(system_mix.variable_number("aux_c")));
				sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_zc")),
  				system_mix.variable(system_mix.variable_number("aux_zc")));
				sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_fc")),
  				system_mix.variable(system_mix.variable_number("aux_fc")));
  			std::cout << "c: " << system_mix.calculate_norm(*system_mix.solution, 0, L2) << " " 
  												<< system_primary.calculate_norm(*system_primary.solution, 0, L2) << std::endl;
				std::cout << "zc: " << system_mix.calculate_norm(*system_mix.solution, 1, L2) << " " 
  												<< system_primary.calculate_norm(*system_primary.solution, 1, L2) << std::endl;
				std::cout << "fc: " << system_mix.calculate_norm(*system_mix.solution, 2, L2) << " " 
  												<< system_primary.calculate_norm(*system_primary.solution, 2, L2) << std::endl;
				std::cout << "aux_c: " << system_mix.calculate_norm(*system_mix.solution, 3, L2) << " " 
  												<< system_aux.calculate_norm(*system_aux.solution, 0, L2) << std::endl;
				std::cout << "aux_zc: " << system_mix.calculate_norm(*system_mix.solution, 4, L2) << " " 
  												<< system_aux.calculate_norm(*system_aux.solution, 1, L2) << std::endl;
				std::cout << "aux_fc: " << system_mix.calculate_norm(*system_mix.solution, 5, L2) << " " 
  												<< system_aux.calculate_norm(*system_aux.solution, 2, L2) << std::endl;
  			std::cout << "Overall: " << system_mix.calculate_norm(*system_mix.solution, L2) << std::endl;  
        system_mix.postprocess();
        
        //DEBUG
        std::cout << " M_HF(psiLF): " << std::setprecision(17) << system_mix.get_MHF_psiLF() << "\n";
  			std::cout << " I(psiLF): " << std::setprecision(17) << system_mix.get_MLF_psiLF() << "\n";
      }

    // Advance to the next timestep in a transient problem
    system_primary.time_solver->advance_timestep();

#ifdef LIBMESH_HAVE_EXODUS_API
    // Write out this timestep if we're requested to
    if ((t_step+1)%write_interval == 0)
      {
        //std::ostringstream file_name;

        // We write the file in the ExodusII format.
        //file_name << "out_"
        //          << std::setw(3)
        //          << std::setfill('0')
        //          << std::right
        //          << t_step+1
        //          << ".e";

        //ExodusII_IO(mesh).write_timestep(file_name.str(),
        ExodusII_IO(mesh).write_timestep("psiLF.exo",
                                         equation_systems_mix,
                                         1, /* This number indicates how many time steps
                                               are being written to the file */
                                         system_primary.time);
     		mesh.write("psiLF_mesh.xda");
     		equation_systems_mix.write("psiLF.xda", WRITE, EquationSystems::WRITE_DATA | 
               EquationSystems::WRITE_ADDITIONAL_DATA);
        equation_systems.write("psiLF_split.xda", WRITE, EquationSystems::WRITE_DATA | 
		     	EquationSystems::WRITE_ADDITIONAL_DATA);
      }
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
  }
  
  // All done.
  return 0;
  
} //end main
