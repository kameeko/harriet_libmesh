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
#include "libmesh/uniform_refinement_estimator.h"
#include "libmesh/getpot.h"

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
  const unsigned int dim                = 3;
  
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  
  //create mesh
  MeshTools::Generation::build_cube(mesh, 
                                    nx, ny, nz, 
                                    497150.0, 501750.0, 
                                    537350.0, 540650.0, 
                                    0.0, 100.0, 
                                    HEX27);
  
  // Print information about the mesh to the screen.
  mesh.print_info();
  
  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  //name system
  ContamTransSys & system = 
  	equation_systems.add_system<ContamTransSys>("ContamTrans");
  	
  //solve as steady or transient
  if(transient)
    system.time_solver = AutoPtr<TimeSolver>(new EulerSolver(system));
  else{
    system.time_solver = AutoPtr<TimeSolver>(new SteadySolver(system));
    libmesh_assert_equal_to (n_timesteps, 1);
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
  NewtonSolver *solver = new NewtonSolver(system); //
  system.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver); //
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

  // Print information about the system to the screen.
  equation_systems.print_info();
  
  for (unsigned int t_step=0; t_step != n_timesteps; ++t_step){
  
    std::cout << "\n\nSolving time step " << t_step << ", time = "
              << system.time << std::endl;
    
    system.solve();
    system.postprocess();
    
    // Advance to the next timestep in a transient problem
    system.time_solver->advance_timestep();

#ifdef LIBMESH_HAVE_EXODUS_API
    // Write out this timestep if we're requested to
    if ((t_step+1)%write_interval == 0){
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
