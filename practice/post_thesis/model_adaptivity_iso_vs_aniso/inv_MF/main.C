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
#include "libmesh/direct_solution_transfer.h"

//local includes
#include "convdiff_mprime.h"
#include "convdiff_primary.h"
#include "convdiff_aux.h"
#include "convdiff_sadj_primary.h"
#include "convdiff_sadj_aux.h"
#include "initial.h"

#include "libmesh/sparse_matrix.h" //DEBUG

int main(int argc, char** argv){

  //for record-keeping
  std::cout << "Running: " << argv[0];
  for (int i=1; i<argc; i++)
    std::cout << " " << argv[i];
  std::cout << std::endl << std::endl;

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("fem_system_params.in");
  unsigned int n_timesteps             = infile("n_timesteps", 1);
  const int nx                         = infile("nx",100);
  const int ny                         = infile("ny",100);
  const int nz                         = infile("nz",100);
  GetPot infileForMesh("contamTrans.in");
  bool solveMF = infileForMesh("solveMF",true); //if false, then solving for LF to read in later...
  bool solveHF = infileForMesh("solveHF",false); 
  bool estErr = infileForMesh("estimate_error",false);

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());
  Mesh mesh2(init.comm()); 
  
  //read in mesh (with MF divvy)
  if(solveMF){
    std::string find_mesh_here = infileForMesh("divided_mesh","divvy1.exo");
    mesh.read(find_mesh_here);
    MeshTools::Generation::build_cube(mesh2, nx, ny, nz, 0., 2300., 0., 1650., 0., 100., HEX27);
  }else{
    MeshTools::Generation::build_cube(mesh, nx, ny, nz, 0., 2300., 0., 1650., 0., 100., HEX27);
    if(solveHF){
      MeshBase::element_iterator       elem_it  = mesh.elements_begin();
      const MeshBase::element_iterator elem_end = mesh.elements_end();
      for (; elem_it != elem_end; ++elem_it){
        Elem* elem = *elem_it;
        elem->subdomain_id() = 1;
      }
    }
    MeshTools::Generation::build_cube(mesh2, nx, ny, nz, 0., 2300., 0., 1650., 0., 100., HEX27);
    std::cout << "\n\nAaaahhhh are you having LF be iso or aniso?" << std::endl;
  }
  
  mesh.print_info(); //DEBUG
  
  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  EquationSystems equation_systems_mix(mesh2);
  
  //name system
  ConvDiff_PrimarySys & system_primary = 
    equation_systems.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys"); //for primary variables
  ConvDiff_AuxSys & system_aux = 
    equation_systems.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys"); //for auxiliary variables
  ConvDiff_MprimeSys & system_mix = 
    equation_systems_mix.add_system<ConvDiff_MprimeSys>("Diff_ConvDiff_MprimeSys"); //for superadj
  ConvDiff_PrimarySadjSys & system_sadj_primary = 
    equation_systems.add_system<ConvDiff_PrimarySadjSys>("ConvDiff_PrimarySadjSys"); //for split superadj
  ConvDiff_AuxSadjSys & system_sadj_aux = 
    equation_systems.add_system<ConvDiff_AuxSadjSys>("ConvDiff_AuxSadjSys"); //for split superadj
  
  //steady-state problem	
 	system_primary.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_primary));
  system_aux.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_aux));
  system_mix.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_mix));
  system_sadj_primary.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_sadj_primary));
  system_sadj_aux.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_sadj_aux));
  
  // Initialize the system
  //equation_systems.init ();
  equation_systems_mix.init();
  
  //initial guess  
  if(solveMF || solveHF){
    std::string find_psiLF_here = infileForMesh("psiLF_file","psiLF.xda");
    std::cout << "Looking for psiLF at: " << find_psiLF_here << "\n\n";
    equation_systems.read(find_psiLF_here, READ,
			  EquationSystems::READ_HEADER |
			  EquationSystems::READ_DATA |
			  EquationSystems::READ_ADDITIONAL_DATA);
  }else{
    equation_systems.init ();
	  read_initial_parameters();
    system_primary.project_solution(initial_value, initial_grad,
                            equation_systems.parameters);
    finish_initialization();
  }
  
  // And the nonlinear solver options
  NewtonSolver *solver_sadj_primary = new NewtonSolver(system_sadj_primary); 
  system_sadj_primary.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_sadj_primary); 
  solver_sadj_primary->quiet = infile("solver_quiet", true);
  solver_sadj_primary->verbose = !solver_sadj_primary->quiet;
  solver_sadj_primary->max_nonlinear_iterations =
    infile("max_nonlinear_iterations", 15);
  solver_sadj_primary->relative_step_tolerance =
    infile("relative_step_tolerance", 1.e-3);
  solver_sadj_primary->relative_residual_tolerance =
    infile("relative_residual_tolerance", 0.0);
  solver_sadj_primary->absolute_residual_tolerance =
    infile("absolute_residual_tolerance", 0.0);
  NewtonSolver *solver_sadj_aux = new NewtonSolver(system_sadj_aux); 
  system_sadj_aux.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_sadj_aux); 
  solver_sadj_aux->quiet = infile("solver_quiet", true);
  solver_sadj_aux->verbose = !solver_sadj_aux->quiet;
  solver_sadj_aux->max_nonlinear_iterations =
    infile("max_nonlinear_iterations", 15);
  solver_sadj_aux->relative_step_tolerance =
    infile("relative_step_tolerance", 1.e-3);
  solver_sadj_aux->relative_residual_tolerance =
    infile("relative_residual_tolerance", 0.0);
  solver_sadj_aux->absolute_residual_tolerance =
    infile("absolute_residual_tolerance", 0.0);
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
    
  //linear solver options
  solver_primary->max_linear_iterations       = infile("max_linear_iterations",10000);
  solver_primary->initial_linear_tolerance    = infile("initial_linear_tolerance",1.e-13);
  solver_primary->minimum_linear_tolerance    = infile("minimum_linear_tolerance",1.e-13);
  solver_primary->linear_tolerance_multiplier = infile("linear_tolerance_multiplier",1.e-3);
  solver_aux->max_linear_iterations           = infile("max_linear_iterations",10000);
  solver_aux->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver_aux->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver_aux->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);
  solver_sadj_primary->max_linear_iterations       = infile("max_linear_iterations",10000);
  solver_sadj_primary->initial_linear_tolerance    = infile("initial_linear_tolerance",1.e-13);
  solver_sadj_primary->minimum_linear_tolerance    = infile("minimum_linear_tolerance",1.e-13);
  solver_sadj_primary->linear_tolerance_multiplier = infile("linear_tolerance_multiplier",1.e-3);
  solver_sadj_aux->max_linear_iterations           = infile("max_linear_iterations",10000);
  solver_sadj_aux->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver_sadj_aux->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver_sadj_aux->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info(); //DEBUG
  
  clock_t begin_inv = std::clock();
  system_primary.solve();
  system_primary.clearQoI();
  clock_t end_inv = std::clock();
  //system_primary.matrix->print_matlab("eep.mat"); //DEBUG
  clock_t begin_err_est = std::clock();
  if(estErr){
    std::cout << "\n End primary solve, begin auxiliary solve..." << std::endl;
    system_aux.solve();
    std::cout << "\n End auxiliary solve..." << std::endl;
  }
  clock_t end_aux = std::clock();

  system_primary.postprocess();
  std::cout << "QoI: " << std::setprecision(17) << system_primary.getQoI() << std::endl;

  clock_t begin_sadj = std::clock();
  clock_t end_sadj = std::clock();
  if(estErr){
    system_aux.postprocess();
    
    system_sadj_primary.set_c_vals(system_primary.get_c_vals());
    system_sadj_aux.set_auxc_vals(system_aux.get_auxc_vals());

    equation_systems_mix.reinit();
    //combine primary and auxiliary variables into psi
    DirectSolutionTransfer sol_transfer(init.comm()); 
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_c")),
      system_mix.variable(system_mix.variable_number("aux_c")));
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_zc")),
      system_mix.variable(system_mix.variable_number("aux_zc")));
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_fc")),
      system_mix.variable(system_mix.variable_number("aux_fc")));
    AutoPtr<NumericVector<Number> > just_aux = system_mix.solution->clone();
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("c")),
      system_mix.variable(system_mix.variable_number("c")));
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("zc")),
      system_mix.variable(system_mix.variable_number("zc")));
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("fc")),
      system_mix.variable(system_mix.variable_number("fc")));
      
    system_mix.assemble(); //calculate residual to correspond to solution
    begin_sadj = std::clock();
    std::cout << "\n Begin primary super-adjoint solve...\n" << std::endl;
    system_sadj_primary.solve();
    std::cout << "\n End primary super-adjoint solve, begin auxiliary super-adjoint solve...\n" << std::endl;
    system_sadj_aux.solve();
    std::cout << "\n End auxiliary super-adjoint solve...\n" << std::endl;
    end_sadj = std::clock();

    const std::string & adjoint_solution0_name = "adjoint_solution0";
    system_mix.add_vector(adjoint_solution0_name, false, GHOSTED);
    system_mix.set_vector_as_adjoint(adjoint_solution0_name,0);
    NumericVector<Number> &eep = system_mix.add_adjoint_rhs(0);
    system_mix.set_adjoint_already_solved(true);
    NumericVector<Number> &dual_sol = system_mix.get_adjoint_solution(0);
    NumericVector<Number> &primal_sol = *system_mix.solution;
    dual_sol.swap(primal_sol);
    sol_transfer.transfer(system_sadj_aux.variable(system_sadj_aux.variable_number("sadj_aux_c")),
      system_mix.variable(system_mix.variable_number("aux_c")));
    sol_transfer.transfer(system_sadj_aux.variable(system_sadj_aux.variable_number("sadj_aux_zc")),
      system_mix.variable(system_mix.variable_number("aux_zc")));
    sol_transfer.transfer(system_sadj_aux.variable(system_sadj_aux.variable_number("sadj_aux_fc")),
      system_mix.variable(system_mix.variable_number("aux_fc")));
    sol_transfer.transfer(system_sadj_primary.variable(system_sadj_primary.variable_number("sadj_c")),
      system_mix.variable(system_mix.variable_number("c")));
    sol_transfer.transfer(system_sadj_primary.variable(system_sadj_primary.variable_number("sadj_zc")),
      system_mix.variable(system_mix.variable_number("zc")));
    sol_transfer.transfer(system_sadj_primary.variable(system_sadj_primary.variable_number("sadj_fc")),
      system_mix.variable(system_mix.variable_number("fc")));

    dual_sol.swap(primal_sol);
    NumericVector<Number> &dual_solution = system_mix.get_adjoint_solution(0);
  
    AutoPtr<NumericVector<Number> > adjresid = system_mix.solution->zero_clone();
    adjresid->pointwise_mult(*system_mix.rhs,dual_solution); 
    adjresid->scale(-0.5);
    std::cout << "\n -0.5*M'_HF(psiLF)(superadj): " << adjresid->sum() << std::endl; //DEBUG
  
    //LprimeHF(psiLF)
    AutoPtr<NumericVector<Number> > LprimeHF_psiLF = system_mix.solution->zero_clone();
    LprimeHF_psiLF->pointwise_mult(*system_mix.rhs,*just_aux);
    std::cout << " L'_HF(psiLF): " << LprimeHF_psiLF->sum() << std::endl; //DEBUG
  

    //QoI and error estimate
    std::cout << "QoI Error estimate: " << std::setprecision(17) 
        << adjresid->sum()+LprimeHF_psiLF->sum() << std::endl; 
      
    double relError = fabs((adjresid->sum()+LprimeHF_psiLF->sum())/system_primary.getQoI());
  
    //print out information
    std::cout << "Estimated absolute relative qoi error: " << relError << std::endl << std::endl;
    std::cout << "Estimated HF QoI: " << std::setprecision(17) << system_primary.getQoI()+adjresid->sum()+LprimeHF_psiLF->sum() << std::endl;
  }
  clock_t end = clock();
  std::cout << "Time for inverse problem: " << double(end_inv-begin_inv)/CLOCKS_PER_SEC << " seconds..." << std::endl;
  std::cout << "Time for extra error estimate bits: " << double(end-begin_err_est)/CLOCKS_PER_SEC << " seconds..." << std::endl;
  std::cout << "    Time to get auxiliary problems: " << double(end_aux-begin_err_est)/CLOCKS_PER_SEC << " seconds..." << std::endl;
  std::cout << "    Time to get superadjoint: " << double(end_sadj-begin_sadj)/CLOCKS_PER_SEC << " seconds...\n" << std::endl;
  
  if(!solveMF && !solveHF)
    equation_systems.write("psiLF.xda", WRITE, EquationSystems::WRITE_DATA | 
               EquationSystems::WRITE_ADDITIONAL_DATA);

  // All done.
  return 0;
  
} //end main
