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
  //const unsigned int max_r_steps        = infile("max_r_steps", 3);
  //const unsigned int max_r_level        = infile("max_r_level", 3);
  //const Real refine_percentage          = infile("refine_percentage", 0.1);
  //const Real coarsen_percentage         = infile("coarsen_percentage", 0.0);
  //const std::string indicator_type      = infile("indicator_type", "kelly");
  //const bool write_error                = infile("write_error",false);
  //const bool flag_by_elem_frac          = infile("flag_by_elem_frac",true);
  
  GetPot infileForMesh("contamTrans.in");
  bool doContinuation                  = infileForMesh("do_continuation",false);
  Real target_R = infileForMesh("reaction_rate",1.e-3);
      
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
/*    MeshTools::Generation::build_cube(mesh, 
                                      nx, ny, nz, 
                                      498300.0, 500600.0, 
                                      538175.0, 539825.0, 
                                      0.0, 100.0, 
                                      HEX27); *///"halved" domain
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
  solver->require_residual_reduction = infile("require_residual_reduction",true);    

  //...and the linear solver options
  solver->max_linear_iterations           = infile("max_linear_iterations", 10000);
  solver->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);
    
  // Mesh Refinement object - to test effect of constant refined mesh (not refined at every timestep)
  //MeshRefinement mesh_refinement(mesh);
  //mesh_refinement.refine_fraction() = refine_percentage;
  //mesh_refinement.coarsen_fraction() = coarsen_percentage;
  //mesh_refinement.max_h_level() = max_r_level;

  // Print information about the system to the screen.
  equation_systems.print_info();
  
  ExodusII_IO exodusIO = ExodusII_IO(mesh); //for writing multiple timesteps to one file
  
  if(!doContinuation){
    system.set_R(target_R);
    system.solve();
    system.postprocess();
    //print out solution minimum to debug oscillations/non-physicality
    NumericVector<Number> &primal_solution = *system.solution;
    double sol_min = primal_solution.min();
    std::cout << "Solution min: " << sol_min << std::endl;
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
      
      //print out solution minimum to debug oscillations/non-physicality
      NumericVector<Number> &primal_solution = *system.solution;
      double sol_min = primal_solution.min();
      std::cout << "Solution min: " << sol_min << std::endl;
    }
    system.postprocess();
  } //end continuation type switch
    
    
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
