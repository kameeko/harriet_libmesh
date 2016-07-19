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

//for debugging
#include "libmesh/sparse_matrix.h"

int main(int argc, char** argv){

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("general.in");
  const bool transient                  = infile("transient", false);
  const Real deltat                     = infile("deltat", 0.005);
  unsigned int n_timesteps              = infile("n_timesteps", 20);
  const int nx                          = infile("nx",100);
  const int ny                          = infile("ny",100);
  const int nz                          = infile("nz",100);
  bool do_2D                            = infile("do_2D",true);
  bool do_square                        = infile("do_square",true);
      
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  
  //create mesh
  unsigned int dim;
  if(do_2D){ 
    dim = 2;
    if(do_square)
      MeshTools::Generation::build_square(mesh, nx, nz, 0.0, 100.0, 0.0, 100.0, QUAD9); //vertical slice
    else{
      MeshTools::Generation::build_square(mesh, nx, nz, 0.0, 4600.0, 0.0, 100.0, QUAD9); //vertical slice
      //MeshTools::Generation::build_square(mesh, nx, ny, 0.0, 4600.0, 0.0, 3300.0, QUAD9); //horizontal slice
    }
  }else{
    dim = 3;
    MeshTools::Generation::build_cube(mesh, 
                                      nx, ny, nz, 
                                      0.0, 4600.0, 
                                      0.0, 3300.0, 
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
  
  //initial conditions/guess
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
  solver->require_residual_reduction = infile("require_residual_reduction",true);    

  //...and the linear solver options
  solver->max_linear_iterations           = infile("max_linear_iterations", 10000);
  solver->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info();
  
  ExodusII_IO exodusIO = ExodusII_IO(mesh); //for writing multiple timesteps to one file

  system.solve();
  system.postprocess();
  
  Number QoI_computed = system.get_QoI_value("computed", 0);
  std::cout<< "Computed QoI is " << std::setprecision(17) << QoI_computed << std::endl;
  
  std::ostringstream Jfile_name;
  Jfile_name << "J.dat";
  std::ofstream outputJ(Jfile_name.str());
  system.matrix->print(outputJ);
  outputJ.close();
    
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
