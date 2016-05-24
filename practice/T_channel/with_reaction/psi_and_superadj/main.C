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
#include "libmesh/enum_xdr_mode.h"

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"

#include "convdiff_mprime.h"
#include "convdiff_primary.h"
#include "convdiff_aux.h"

#include "libmesh/dof_map.h" //alternate error breakdown
#include "libmesh/direct_solution_transfer.h"

//FOR DEBUGGING
#include "libmesh/sparse_matrix.h" //DEBUG
#include "libmesh/gmv_io.h" //DEBUG

// The main program
int main(int argc, char** argv)
{

  // Initialize libMesh
  LibMeshInit init(argc, argv);
	
  // Parameters
  GetPot infile("fem_system_params.in");
  const bool transient                  = infile("transient", false);
  unsigned int n_timesteps              = infile("n_timesteps", 1);
  GetPot infileForMesh("convdiff_mprime.in");
  std::string find_mesh_here            = infileForMesh("divided_mesh","mesh.exo");
  bool doContinuation                   = infileForMesh("do_continuation",false);

  Mesh mesh(init.comm());
  Mesh mesh2(init.comm()); 
  mesh.read(find_mesh_here);
  mesh2.read(find_mesh_here); 
  
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
		
	//steady-state problem	
 	system_primary.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_primary));
  system_aux.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_aux));
  system_mix.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_mix));
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
	equation_systems.init ();
	equation_systems_mix.init();
	
	//nonlinear solver options
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
  solver_primary->max_linear_iterations       = infile("max_linear_iterations", 50000);
  solver_primary->max_linear_iterations       = infile("max_linear_iterations",10000);
  solver_primary->initial_linear_tolerance    = infile("initial_linear_tolerance",1.e-13);
  solver_primary->minimum_linear_tolerance    = infile("minimum_linear_tolerance",1.e-13);
  solver_primary->linear_tolerance_multiplier = infile("linear_tolerance_multiplier",1.e-3);
 	solver_aux->max_linear_iterations           = infile("max_linear_iterations", 50000);
  solver_aux->max_linear_iterations           = infile("max_linear_iterations",10000);
  solver_aux->initial_linear_tolerance        = infile("initial_linear_tolerance",1.e-13);
  solver_aux->minimum_linear_tolerance        = infile("minimum_linear_tolerance",1.e-13);
  solver_aux->linear_tolerance_multiplier     = infile("linear_tolerance_multiplier",1.e-3);
  
  equation_systems.print_info(); //DEBUG
  
//ITERATE THROUGH REFINEMENTS  
  
  system_primary.solve();
  std::cout << "\n End primary solve, begin auxiliary solve..." << std::endl;
  system_aux.solve();
  std::cout << "\n End auxiliary solve..." << std::endl;
  
std::clock_t start = std::clock(); //DEBUG
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
	
double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC; //DEBUG
std::cout<<"\n Solution transfer time: "<< duration <<'\n';	 //DEBUG
std::cout << "c: " << system_mix.calculate_norm(*system_mix.solution, 0, L2) << " " 
									<< system_primary.calculate_norm(*system_primary.solution, 0, L2) << std::endl; //DEBUG
std::cout << "zc: " << system_mix.calculate_norm(*system_mix.solution, 1, L2) << " " 
									<< system_primary.calculate_norm(*system_primary.solution, 1, L2) << std::endl; //DEBUG
std::cout << "fc: " << system_mix.calculate_norm(*system_mix.solution, 2, L2) << " " 
									<< system_primary.calculate_norm(*system_primary.solution, 2, L2) << std::endl; //DEBUG
std::cout << "aux_c: " << system_mix.calculate_norm(*system_mix.solution, 3, L2) << " " 
									<< system_aux.calculate_norm(*system_aux.solution, 0, L2) << std::endl; //DEBUG
std::cout << "aux_zc: " << system_mix.calculate_norm(*system_mix.solution, 4, L2) << " " 
									<< system_aux.calculate_norm(*system_aux.solution, 1, L2) << std::endl; //DEBUG
std::cout << "aux_fc: " << system_mix.calculate_norm(*system_mix.solution, 5, L2) << " " 
									<< system_aux.calculate_norm(*system_aux.solution, 2, L2) << std::endl; //DEBUG

  //where to put postprocess? what things really need to be called from postprocess?
  
  //IS THIS NECESSARY? DOESN'T FEED INTO ANYTHING ELSE...
  //QoISet qois;
  //std::vector<unsigned int> qoi_indices;
  //qoi_indices.push_back(0);
  //qois.add_indices(qoi_indices);
  //qois.set_weight(0, 1.0);

  system_mix.assemble_qoi_sides = true; //QoI doesn't involve sides
  
  std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve start ~*~*~*~*~*~*~*~*~\n" << std::endl;
  std::pair<unsigned int, Real> adjsolve = system_mix.adjoint_solve();
  std::cout << "number of iterations to solve adjoint: " << adjsolve.first << std::endl;
  std::cout << "final residual of adjoint solve: " << adjsolve.second << std::endl;
	std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve end ~*~*~*~*~*~*~*~*~" << std::endl;
	
	NumericVector<Number> &dual_solution = system_mix.get_adjoint_solution(0);
  NumericVector<Number> &primal_solution = *system_mix.solution;

  system_mix.assemble(); //calculate residual to correspond to solution
  
  //adjoint-weighted residual
  AutoPtr<NumericVector<Number> > adjresid = system_mix.solution->clone();
	adjresid->zero();
	adjresid->pointwise_mult(*system_mix.rhs,dual_solution); 
	adjresid->scale(-0.5);
	std::cout << "\n -0.5*M'_HF(psiLF)(superadj): " << adjresid->sum() << std::endl; //DEBUG
  
  //LprimeHF(psiLF)
  AutoPtr<NumericVector<Number> > LprimeHF_psiLF = system_mix.solution->clone();
  LprimeHF_psiLF->zero();
  LprimeHF_psiLF->pointwise_mult(*system_mix.rhs,*just_aux);
  std::cout << " L'_HF(psiLF): " << LprimeHF_psiLF->sum() << std::endl; //DEBUG
  
  //QoI and error estimate
  system_primary.postprocess();
  std::cout << "QoI: " << std::setprecision(17) << system_primary.getQoI() << std::endl; //MAKE ME, CLEAN UP POSTPROCESS IN MIX
  std::cout << "QoI Error estimate: " << std::setprecision(17) 
	    << adjresid->sum()+LprimeHF_psiLF->sum() << std::endl; 
  
  //output if last iteration
  std::string write_error_basis_blame = 
    infileForMesh("error_est_output_file_basis_blame", "error_est_breakdown_basis_blame.dat");
  std::ofstream output2(write_error_basis_blame);
  for(unsigned int i = 0 ; i < adjresid->size(); i++){
    if(output2.is_open())
      output2 << (*adjresid)(i) + (*LprimeHF_psiLF)(i) << "\n";
  }
  output2.close();
  //DOF maps and such to help visualize
  std::ofstream output_global_dof("global_dof_map.dat");
	for(unsigned int i = 0 ; i < system_mix.get_mesh().n_elem(); i++){
	  std::vector< dof_id_type > di;
	  system_mix.get_dof_map().dof_indices(system_mix.get_mesh().elem(i), di);
		if(output_global_dof.is_open()){
			output_global_dof << i << " ";
			for(unsigned int j = 0; j < di.size(); j++)
			  output_global_dof << di[j] << " ";
			output_global_dof << "\n";
		}
	}
	output_global_dof.close();
	std::ofstream output_var_dof("var_dof_map.dat");
	for(unsigned int var_num = 0; var_num < system_mix.get_dof_map().n_variables(); var_num++){
	  std::vector< dof_id_type > di;
	  system_mix.get_dof_map().local_variable_indices(di, mesh2, var_num);
	  if(output_var_dof.is_open()){
			output_var_dof << var_num << " ";
			for(unsigned int j = 0; j < di.size(); j++)
			  output_var_dof << di[j] << " ";
			output_var_dof << "\n";
		}
	}
	output_var_dof.close();
	std::ofstream output_elem_cent("elem_centroids.dat");
	for(unsigned int i = 0 ; i < system_mix.get_mesh().n_elem(); i++){
		Point elem_cent = system_mix.get_mesh().elem(i)->centroid();
		if(output_elem_cent.is_open()){
			output_elem_cent << elem_cent(0) << " " << elem_cent(1) << "\n";
		}
	}
	output_elem_cent.close();


//TRY SOLVING SUPERADJ AS SPLIT FORWARD AFTER FIRST ROUND OF INTEGRATION? if so use different (not-newton) solver that knows it should be linear?
  //put in same eq sys as primary and aux so you can grab values to linearize about?
} //end main
