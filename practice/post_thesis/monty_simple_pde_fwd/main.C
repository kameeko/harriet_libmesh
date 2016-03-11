//foward run for simple contaminant transport model

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
#include "libmesh/patch_recovery_error_estimator.h"
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/getpot.h"
#include "libmesh/tecplot_io.h"

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"
#include "libmesh/euler_solver.h"

//local includes
#include "contamTrans_sys.h"
#include "initial.h"

int main(int argc, char** argv){

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("fem_system_params.in");
  const bool transient                  = infile("transient", true);
  const Real deltat                     = infile("deltat", 0.005);
  unsigned int n_timesteps              = infile("n_timesteps", 20);
  const int nx                          = infile("nx",100);
  const int ny                          = infile("ny",100);
  const int nz                          = infile("nz",100);
  //const unsigned int dim                = 3;
  const unsigned int max_r_steps        = infile("max_r_steps", 3);
  const unsigned int max_r_level        = infile("max_r_level", 3);
  const Real refine_percentage          = infile("refine_percentage", 0.5);
  const Real coarsen_percentage         = infile("coarsen_percentage", 0.5);
  const std::string indicator_type      = infile("indicator_type", "kelly");
      
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  
  //create mesh
  unsigned int dim;
  if(nz == 0){ //to check if oscillations happen in 2D as well...
    dim = 2;
    MeshTools::Generation::build_square(mesh, nx, ny, 497150.0, 501750.0, 537350.0, 540650.0, QUAD9);
  }else{
    dim = 3;
    MeshTools::Generation::build_cube(mesh, 
                                      nx, ny, nz, 
                                      497150.0, 501750.0, 
                                      537350.0, 540650.0, 
                                      0.0, 100.0, 
                                      HEX27);
  }
  
  // Print information about the mesh to the screen.
  mesh.print_info();
  
  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  //name system
  ContamTransSys & system = 
  	equation_systems.add_system<ContamTransSys>("ContamTrans");
  	
  //solve as steady or transient
  if(transient){
    system.time_solver = AutoPtr<TimeSolver>(new EulerSolver(system)); //backward Euler
    std::cout << "\n\nAaahhh time derivative term doesn't yet include porosity!\n" << std::endl;
  }
  else{
    system.time_solver = AutoPtr<TimeSolver>(new SteadySolver(system));
    libmesh_assert_equal_to (n_timesteps, 1); //this doesn't seem to work?
  }
  
  // Initialize the system
  equation_systems.init ();
  
  //initial conditions
  read_initial_parameters();
  system.project_solution(initial_value, initial_grad,
                          equation_systems.parameters);
  finish_initialization();

  // Set the time stepping options...
  system.deltat = deltat;
  
  //...and the nonlinear solver options...
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
    
  //...and the linear solver options
  solver->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);
    
  // Mesh Refinement object
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.refine_fraction() = refine_percentage;
  mesh_refinement.coarsen_fraction() = coarsen_percentage;
  mesh_refinement.max_h_level() = max_r_level;

  // Print information about the system to the screen.
  equation_systems.print_info();
  
  ExodusII_IO exodusIO = ExodusII_IO(mesh); //for writing multiple timesteps to one file
  
  for (unsigned int r_step=0; r_step<max_r_steps; r_step++)
    {
      std::cout << "\nBeginning Solve " << r_step+1 << std::endl;
      
      for (unsigned int t_step=0; t_step != n_timesteps; ++t_step)
        {
      
          std::cout << "\n\nSolving time step " << t_step << ", time = "
                    << system.time << std::endl;
          
          system.solve();
          system.postprocess();
          
          //print out solution minimum to debug oscillations
          NumericVector<Number> &primal_solution = *system.solution;
          std::cout << "Solution min: " << primal_solution.min() << std::endl;
          
          // Advance to the next timestep in a transient problem
          system.time_solver->advance_timestep();
   
      } //end stepping through time loop
                
      std::cout << "\n  Refining the mesh..." << std::endl;
      
      // The \p ErrorVector is a particular \p StatisticsVector
      // for computing error information on a finite element mesh.
      ErrorVector error;
      
      if (indicator_type == "patch")
        {
          // The patch recovery estimator should give a
          // good estimate of the solution interpolation
          // error.
          PatchRecoveryErrorEstimator error_estimator;

          error_estimator.estimate_error (system, error);
        }
      else if (indicator_type == "kelly")
        {

          // The Kelly error estimator is based on
          // an error bound for the Poisson problem
          // on linear elements, but is useful for
          // driving adaptive refinement in many problems
          KellyErrorEstimator error_estimator;

          error_estimator.estimate_error (system, error);
        }
      
      // Write out the error distribution
      std::ostringstream ss;
      ss << r_step;
#ifdef LIBMESH_HAVE_EXODUS_API
      std::string error_output = "error_"+ss.str()+".e";
#else
      std::string error_output = "error_"+ss.str()+".gmv";
#endif
      error.plot_error( error_output, mesh );
      
      // This takes the error in \p error and decides which elements
      // will be coarsened or refined.  Any element within 20% of the
      // maximum error on any element will be refined, and any
      // element within 10% of the minimum error on any element might
      // be coarsened. Note that the elements flagged for refinement
      // will be refined, but those flagged for coarsening _might_ be
      // coarsened.
      mesh_refinement.flag_elements_by_error_fraction (error);
      
      // This call actually refines and coarsens the flagged
      // elements.
      mesh_refinement.refine_and_coarsen_elements();
      
      // This call reinitializes the \p EquationSystems object for
      // the newly refined mesh.  One of the steps in the
      // reinitialization is projecting the \p solution,
      // \p old_solution, etc... vectors from the old mesh to
      // the current one.
      equation_systems.reinit ();
      
      std::cout << "System has: " << equation_systems.n_active_dofs()
                << " degrees of freedom."
                << std::endl;
      
    } //end refinement loop
    
  //use that final refinement
  for (unsigned int t_step=0; t_step != n_timesteps; ++t_step)
    {
  
      std::cout << "\n\nSolving time step " << t_step << ", time = "
                << system.time << std::endl;
      
      system.solve();
      system.postprocess();
      
      //print out solution minimum to debug oscillations
      NumericVector<Number> &primal_solution = *system.solution;
      std::cout << "Solution min: " << primal_solution.min() << std::endl;
      
      // Advance to the next timestep in a transient problem
      system.time_solver->advance_timestep();

    } //end stepping through time loop
    
    
#ifdef LIBMESH_HAVE_EXODUS_API
  for (unsigned int t_step=0; t_step != n_timesteps; ++t_step)
    {
      // Write out this timestep if we're requested to
      if ((t_step+1)%write_interval == 0)
        {
          std::ostringstream ex_file_name;
          std::ostringstream tplot_file_name;

          // We write the file in the ExodusII format.
          //ex_file_name << "out_"
          //            << std::setw(3)
          //            << std::setfill('0')
          //            << std::right
          //            << t_step+1
          //            << ".e";
                      
          tplot_file_name << "out_"
                      << std::setw(3)
                      << std::setfill('0')
                      << std::right
                      << t_step+1
                      << ".plt";

          //ExodusII_IO(mesh).write_timestep(ex_file_name.str(),
          //                                 equation_systems,
          //                                 1, /* This number indicates how many time steps
          //                                       are being written to the file */
          //                                 system.time);
          exodusIO.write_timestep("output.exo", equation_systems, t_step+1, system.time); //outputs all timesteps in one file
          TecplotIO(mesh).write_equation_systems(tplot_file_name.str(), equation_systems);
        }
    }
#endif // #ifdef LIBMESH_HAVE_EXODUS_API     

  // All done.
  return 0;
  
} //end main
