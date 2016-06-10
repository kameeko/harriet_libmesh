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
#include "libmesh/continuation_system.h"

//local includes
#include "contamTrans_inv.h"
#include "initial.h"

int main(int argc, char** argv){

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("fem_system_params.in");
  unsigned int n_timesteps             = infile("n_timesteps", 1);
  //Real init_dR                         = infile("init_dR",1.e-3);
  const int nx                          = infile("nx",100);
  const int ny                          = infile("ny",100);
  const int nz                          = infile("nz",100);
  GetPot infileForMesh("contamTrans.in");
  bool doContinuation                  = infileForMesh("do_continuation",false);
  bool doArcLengthContinuation         = infileForMesh("do_arclength_continuation",false);
  
  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  
  //create mesh
  unsigned int dim;
  if(nz == 0){ //to check if oscillations happen in 2D as well...
    dim = 2;
    MeshTools::Generation::build_square(mesh, nx, ny, 0., 2300., 0., 1650., QUAD9);
  }else{
    dim = 3;
    MeshTools::Generation::build_cube(mesh, 
                                      nx, ny, nz, 
                                      0., 2300., 
                                      0., 1650., 
                                      0., 100., 
                                      HEX27);
  }

  // Print information about the mesh to the screen.
  mesh.print_info(); //DEBUG

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  //name system
  ContamTransSysInv & system = 
    equation_systems.add_system<ContamTransSysInv>("ContamTransInv");
  //system.min_continuation_parameter = 0.;
  //system.max_continuation_parameter = std::fabs(infileForMesh("reaction_rate",1.e-3));
  
  //steady-state problem	
 	system.time_solver = AutoPtr<TimeSolver>(new SteadySolver(system));
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
  equation_systems.init ();
  
  //initial conditions
  read_initial_parameters();
  system.project_solution(initial_value, initial_grad,
                          equation_systems.parameters);
  finish_initialization();

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
  solver->max_linear_iterations           = infile("max_linear_iterations", 10000);
  solver->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info(); //DEBUG
  
  Real target_R = infileForMesh("reaction_rate",1.e-3);
  
  if(!doContinuation){
    system.set_R(target_R);
    system.solve();
  }else if(doArcLengthContinuation){
  
    std::cout << "\n\nAAAAAAAAAAAAAHHHHHHHHH this arc-length continuation doesn't work yet...\n\n" << std::endl;
    /*
    system.quiet = infile("solver_quiet", true);
    system.set_max_arclength_stepsize(infile("max_ds",1.e2));
      
    //two prior solutions
    system.set_R(0.); 
    system.solve(); //first solution
    system.save_current_solution();
    std::cout << "\n First solve finished..." << std::endl;
    system.set_R(init_dR);
    system.solve(); //second solution
    std::cout << "\n Second solve finished..." << std::endl;
    
    //continuation steps
    //system.set_R(target_R);
    system.continuation_solve(); //doesn't seem to move away first step...what is this even supposed to do?
    std::cout << "\n Continuation solve done..." << std::endl;
    Real Rcurr = system.get_R();
    int iter = 0;
    while(system.get_R() < 0.99*target_R && iter < infile("max_arcsteps",10)){
      system.advance_arcstep();
      system.solve();
      std::cout << "\nR: " << system.get_R() << std::endl;
      std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
      iter += 1;
    }
    */
  }else{ //natural continuation
    std::vector<double> Rsteps;
    if(FILE *fp=fopen("continuation_steps.txt","r")){
			Real value;
			int flag = 1;
			while(flag != -1){
				flag = fscanf(fp,"%lf",&value);
				if(flag != -1){
					Rsteps.push_back(value);
				}
			}
			fclose(fp);
	  }else{
	    std::cout << "\n\n Need to define continuation steps in continuation_steps.txt\n\n" << std::endl;
	  }
	  
	  for(int i = 0; i < Rsteps.size(); i++){
	    double R = Rsteps[i];
	    system.set_R(R);
	    std::cout << "\n\nStarting iteration with R = " << R << "\n\n" << std::endl;
	    system.solve();
	  }
  } //end continuation type switch
  
  system.postprocess();
  Number QoI_computed = system.get_QoI_value("computed", 0);
  std::cout<< "Computed QoI is " << std::setprecision(17) << QoI_computed << std::endl;

  // All done.
  return 0;
  
} //end main
