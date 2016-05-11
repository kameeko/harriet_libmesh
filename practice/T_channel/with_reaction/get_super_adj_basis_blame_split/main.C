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
#include "libmesh/gmv_io.h" //DEBUG

// The systems and solvers we may use
#include "libmesh/diff_solver.h"
#include "libmesh/steady_solver.h"
#include "libmesh/newton_solver.h"
#include "convdiff_mprime.h"
#include "libmesh/sparse_matrix.h" //DEBUG
#include "convdiff_primary.h"
#include "convdiff_aux.h"

#include "libmesh/dof_map.h" //alternate error breakdown
#include "libmesh/direct_solution_transfer.h"

// The main program
int main(int argc, char** argv)
{
  // Initialize libMesh
  LibMeshInit init(argc, argv);
	
  // Parameters
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
  Mesh mesh3(init.comm()); 
  GetPot infileForMesh("convdiff_mprime.in");
  std::string find_mesh_here = infileForMesh("mesh","psiLF_mesh.xda");
  mesh.read(find_mesh_here);
  mesh2.read(find_mesh_here); 
  mesh3.read(find_mesh_here); 
	
  std::cout << "Read in mesh from: " << find_mesh_here << "\n\n";

  // And an object to refine it
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.coarsen_by_parents() = true;
  mesh_refinement.absolute_global_tolerance() = global_tolerance;
  mesh_refinement.nelem_target() = nelem_target;
  mesh_refinement.refine_fraction() = 0.3;
  mesh_refinement.coarsen_fraction() = 0.3;
  mesh_refinement.coarsen_threshold() = 0.1;

  //mesh_refinement.uniformly_refine(coarserefinements);
  
  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  EquationSystems equation_systems_all (mesh3);
  EquationSystems equation_systems2(mesh2);
 
  // Name system
  ConvDiff_PrimarySys & system_primary = 
  	equation_systems.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys");
	ConvDiff_AuxSys & system_aux = 
  	equation_systems.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys");
  ConvDiff_MprimeSys & system =
    equation_systems_all.add_system<ConvDiff_MprimeSys>("Diff_ConvDiff_MprimeSys");
  ConvDiff_MprimeSys & system2 = 
    equation_systems2.add_system<ConvDiff_MprimeSys>("Diff_ConvDiff_MprimeSys");
      
  // Steady-state problem	
  system_primary.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_primary));
  system_aux.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system_aux));
  system.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system));
  system2.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system2));

  // Sanity check that we are indeed solving a steady problem
  libmesh_assert_equal_to (n_timesteps, 1);

  // Read in all the equation systems data from the LF solve (system, solutions, rhs, etc)
  std::string find_psiLF_here = infileForMesh("psiLF_file","psiLF.xda");
  std::cout << "Looking for psiLF at: " << find_psiLF_here << "\n\n";
  
  equation_systems_all.read(find_psiLF_here, READ,
			EquationSystems::READ_HEADER |
			EquationSystems::READ_DATA |
			EquationSystems::READ_ADDITIONAL_DATA);
	equation_systems.read("psiLF_split.xda", READ,
				EquationSystems::READ_HEADER |
				EquationSystems::READ_DATA |
				EquationSystems::READ_ADDITIONAL_DATA);
  
  // Check that the norm of the solution read in is what we expect it to be
  Real readin_L2_primary = system_primary.calculate_norm(*system_primary.solution, L2);  
  std::cout << "Read in solution norm (primary): "<< readin_L2_primary << std::endl << std::endl;

	//DEBUG
  //equation_systems.write("right_back_out.xda", WRITE, EquationSystems::WRITE_DATA |
	//		 EquationSystems::WRITE_ADDITIONAL_DATA);
#ifdef LIBMESH_HAVE_GMV
  //GMVIO(equation_systems.get_mesh()).write_equation_systems(std::string("right_back_out.gmv"), equation_systems);
#endif

  // Initialize the system
  //equation_systems.init ();  //already initialized by read-in

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
  for (unsigned int t_step=0; t_step != n_timesteps; ++t_step)
    {
      // A pretty update message
      std::cout << "\n\nSolving time step " << t_step << ", time = " << system.time << std::endl;
/////////////////////////////////////////////////////////
	  QoISet qois;
	  std::vector<unsigned int> qoi_indices;
	  
	  qoi_indices.push_back(0);
	  qois.add_indices(qoi_indices);
	  
	  qois.set_weight(0, 1.0);

	  system_primary.assemble_qoi_sides = true; //QoI doesn't involve sides
	  system_aux.assemble_qoi_sides = true; //QoI doesn't involve sides
	  
	  std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve start ~*~*~*~*~*~*~*~*~\n" << std::endl;
	  std::pair<unsigned int, Real> adjsolve_primary = system_primary.adjoint_solve();
	  std::cout << "number of iterations to solve adjoint: " << adjsolve_primary.first << std::endl;
	  std::cout << "final residual of adjoint solve: " << adjsolve_primary.second << std::endl;
	  std::pair<unsigned int, Real> adjsolve_aux = system_aux.adjoint_solve();
	  std::cout << "number of iterations to solve adjoint: " << adjsolve_aux.first << std::endl;
	  std::cout << "final residual of adjoint solve: " << adjsolve_aux.second << std::endl;
 		std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve end ~*~*~*~*~*~*~*~*~" << std::endl;
 		
 		NumericVector<Number> &primary_dual_solution = system_primary.get_adjoint_solution(0);
	  NumericVector<Number> &primary_primal_solution = *system_primary.solution;
	  NumericVector<Number> &aux_dual_solution = system_aux.get_adjoint_solution(0);
	  NumericVector<Number> &aux_primal_solution = *system_aux.solution;
	  const std::string & adjoint_solution0_name = "adjoint_solution0";
	  //const std::string & adjoint_rhs0_name = "adjoint_rhs0";
    system.add_vector(adjoint_solution0_name, false, GHOSTED);
    //system.add_vector(adjoint_rhs0_name, false, GHOSTED);
	  //NumericVector<Number> &meep = system.add_vector("combined_adj",false);
	  system.set_vector_as_adjoint(adjoint_solution0_name,0);
	  NumericVector<Number> &eep = system.add_adjoint_rhs(0);
	  system.set_adjoint_already_solved(true);
	  NumericVector<Number> &dual_solution = system.get_adjoint_solution(0);
	  NumericVector<Number> &primal_solution = *system.solution;
	  
	  //combine adjoint
	  primary_primal_solution.swap(primary_dual_solution);
	  aux_primal_solution.swap(aux_dual_solution);
	  primal_solution.swap(dual_solution);
	  std::cout << " Super-adjoint L2 norms: \n" 
	    << system_primary.calculate_norm(primary_primal_solution,0,L2) << "\n" 
	    << system_primary.calculate_norm(primary_primal_solution,1,L2) << "\n" 
	    << system_primary.calculate_norm(primary_primal_solution,2,L2) << "\n" 
	    << system_aux.calculate_norm(aux_primal_solution,0,L2) << "\n" 
	    << system_aux.calculate_norm(aux_primal_solution,1,L2) << "\n" 
	    << system_aux.calculate_norm(aux_primal_solution,2,L2) << "\n" << std::endl;
 		DirectSolutionTransfer sol_transfer1(init.comm());
		sol_transfer1.transfer(system_primary.variable(system_primary.variable_number("c")),
			system.variable(system.variable_number("c")));
		sol_transfer1.transfer(system_primary.variable(system_primary.variable_number("zc")),
			system.variable(system.variable_number("zc")));
		sol_transfer1.transfer(system_primary.variable(system_primary.variable_number("fc")),
			system.variable(system.variable_number("fc")));
		sol_transfer1.transfer(system_aux.variable(system_aux.variable_number("aux_c")),
			system.variable(system.variable_number("aux_c")));
		sol_transfer1.transfer(system_aux.variable(system_aux.variable_number("aux_zc")),
			system.variable(system.variable_number("aux_zc")));
		sol_transfer1.transfer(system_aux.variable(system_aux.variable_number("aux_fc")),
			system.variable(system.variable_number("aux_fc")));
	  std::cout << " Super-adjoint L2 norms: \n" 
	    << system.calculate_norm(primal_solution,0,L2) << "\n" 
	    << system.calculate_norm(primal_solution,1,L2) << "\n" 
	    << system.calculate_norm(primal_solution,2,L2) << "\n" 
	    << system.calculate_norm(primal_solution,3,L2) << "\n" 
	    << system.calculate_norm(primal_solution,4,L2) << "\n" 
	    << system.calculate_norm(primal_solution,5,L2) << "\n" << std::endl;
		primal_solution.swap(dual_solution);
			
	  primal_solution.swap(dual_solution);
	  ExodusII_IO(mesh).write_timestep("super_adjoint.exo",
	                                 equation_systems,
	                                 1, /* This number indicates how many time steps
	                                       are being written to the file */
	                                 system.time);
	  primal_solution.swap(dual_solution);

    system.assemble(); //overwrite residual read in from psiLF solve
        
	  // The total error estimate
	  system.postprocess(); //to compute M_HF(psiLF) and M_LF(psiLF) terms
	  Real QoI_error_estimate = (-0.5*(system.rhs)->dot(dual_solution)) + system.get_MHF_psiLF() - system.get_MLF_psiLF();
	  std::cout << "\n\n 0.5*M'_HF(psiLF)(superadj): " << std::setprecision(17) << 0.5*(system.rhs)->dot(dual_solution) << "\n";
	  std::cout << " M_HF(psiLF): " << std::setprecision(17) << system.get_MHF_psiLF() << "\n";
  	std::cout << " M_LF(psiLF): " << std::setprecision(17) << system.get_MLF_psiLF() << "\n";
	  std::cout << "\n\n Residual L2 norm: " << system.calculate_norm(*system.rhs, L2) << "\n"; 
	  //std::cout << " Residual discrete L2 norm: " << system.calculate_norm(*system.rhs, DISCRETE_L2) << "\n";
	  std::cout << " Super-adjoint L2 norm: " << system.calculate_norm(dual_solution, L2) << "\n";
	  //std::cout << " Super-adjoint discrete L2 norm: " << system.calculate_norm(dual_solution, DISCRETE_L2) << "\n";
	  std::cout << "\n\n QoI error estimate: " << std::setprecision(17) << QoI_error_estimate << "\n\n";
	  
	  //DEBUG
	  std::cout << "\n------------ herp derp ------------" << std::endl;
	  //libMesh::out.precision(16);
	  //dual_solution.print();
	  //system.get_adjoint_rhs().print();

		AutoPtr<NumericVector<Number> > adjresid = system.solution->clone();
		(system.matrix)->vector_mult(*adjresid,system.get_adjoint_solution(0));
		SparseMatrix<Number>& adjmat = *system.matrix; 
		(system.matrix)->get_transpose(adjmat);
		adjmat.vector_mult(*adjresid,system.get_adjoint_solution(0));
		//std::cout << "******************** matrix-superadj product (libmesh) ************************" << std::endl;
		//adjresid->print();
		adjresid->add(-1.0, system.get_adjoint_rhs(0));
		//std::cout << "******************** superadjoint system residual (libmesh) ***********************" << std::endl;
		//adjresid->print();
		std::cout << "\n\nadjoint system residual (discrete L2): " << system.calculate_norm(*adjresid,DISCRETE_L2) << std::endl;
		std::cout << "adjoint system residual (L2, all): " << system.calculate_norm(*adjresid,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 0): " << system.calculate_norm(*adjresid,0,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 1): " << system.calculate_norm(*adjresid,1,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 2): " << system.calculate_norm(*adjresid,2,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 3): " << system.calculate_norm(*adjresid,3,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 4): " << system.calculate_norm(*adjresid,4,L2) << std::endl;
		std::cout << "adjoint system residual (L2, 5): " << system.calculate_norm(*adjresid,5,L2) << std::endl;
		
	  std::cout << "\n------------ herp derp ------------" << std::endl;

	  //element-wise breakdown, outputed as values matched to element centroids; for matlab plotz
	  primal_solution.swap(dual_solution);
	 	system.postprocess(1);
	 	primal_solution.swap(dual_solution);
	 	system.postprocess(2);
	 	std::cout << "\n\n -0.5*M'_HF(psiLF)(superadj): " << std::setprecision(17) << system.get_half_adj_weighted_resid() << "\n";
	 	//primal_solution.swap(dual_solution); //what the heck was this for anyways?
	 	
	 	std::string write_error_here = infileForMesh("error_est_output_file", "error_est_breakdown.dat");
    std::ofstream output(write_error_here);
		for(unsigned int i = 0 ; i < system.get_mesh().n_elem(); i++){
			Point elem_cent = system.get_mesh().elem(i)->centroid();
			if(output.is_open()){
				output << elem_cent(0) << " " << elem_cent(1) << " " 
					<< fabs(system.get_half_adj_weighted_resid(i) + system.get_MHF_psiLF(i) - system.get_MLF_psiLF(i)) << "\n";
				//output << elem_cent(0) << " " << elem_cent(1) << " " 
				//	<< fabs(system.get_half_adj_weighted_resid(i)) << "\n"; //DEBUG
			}
		}
		output.close();
		
		//error-breakdown, with contributions assigned to basis functions instead of elements
		equation_systems2.init();
		DirectSolutionTransfer sol_transfer(init.comm());
		sol_transfer.transfer(system.variable(system.variable_number("aux_c")),
			system2.variable(system2.variable_number("aux_c")));
		sol_transfer.transfer(system.variable(system.variable_number("aux_zc")),
			system2.variable(system2.variable_number("aux_zc")));
		sol_transfer.transfer(system.variable(system.variable_number("aux_fc")),
			system2.variable(system2.variable_number("aux_fc")));
		std::string write_error_basis_blame = infileForMesh("error_est_output_file_basis_blame", "error_est_breakdown_basis_blame.dat");
		AutoPtr<NumericVector<Number> > adjresid_basis_blame = system.solution->clone();
		adjresid_basis_blame->zero();
		adjresid_basis_blame->pointwise_mult(*system.rhs,dual_solution); 
	  std::cout << "\n -0.5*M'_HF(psiLF)(superadj): " << -0.5*adjresid_basis_blame->sum() << std::endl; //check
	  AutoPtr<NumericVector<Number> > LprimeHF_psiLF_basis_blame = system.solution->clone();
	  LprimeHF_psiLF_basis_blame->zero();
	  LprimeHF_psiLF_basis_blame->pointwise_mult(*system.rhs,*system2.solution);
	  std::cout << " L'_HF(psiLF): " << LprimeHF_psiLF_basis_blame->sum() << " vs " 
	    << system.get_MHF_psiLF()-system.get_MLF_psiLF() << std::endl; //check
	  std::cout << " QoI error estimate: " << std::setprecision(17) 
	    << -0.5*adjresid_basis_blame->sum()+LprimeHF_psiLF_basis_blame->sum() << std::endl; //check
	  std::ofstream output2(write_error_basis_blame);
    for(unsigned int i = 0 ; i < adjresid_basis_blame->size(); i++){
	    if(output2.is_open())
	      output2 << -0.5*(*adjresid_basis_blame)(i) + (*LprimeHF_psiLF_basis_blame)(i) << "\n";
	  }
	  output2.close();
	  //DOF maps and such to help visualize
	  std::ofstream output_global_dof("global_dof_map.dat");
		for(unsigned int i = 0 ; i < system.get_mesh().n_elem(); i++){
		  std::vector< dof_id_type > di;
		  system.get_dof_map().dof_indices(system.get_mesh().elem(i), di);
			if(output_global_dof.is_open()){
				output_global_dof << i << " ";
				for(unsigned int j = 0; j < di.size(); j++)
				  output_global_dof << di[j] << " ";
				output_global_dof << "\n";
			}
		}
		output_global_dof.close();
		std::ofstream output_var_dof("var_dof_map.dat");
		for(unsigned int var_num = 0; var_num < system.get_dof_map().n_variables(); var_num++){
		  std::vector< dof_id_type > di;
		  system.get_dof_map().local_variable_indices(di, mesh3, var_num);
		  if(output_var_dof.is_open()){
				output_var_dof << var_num << " ";
				for(unsigned int j = 0; j < di.size(); j++)
				  output_var_dof << di[j] << " ";
				output_var_dof << "\n";
			}
		}
		output_var_dof.close();
		std::ofstream output_elem_cent("elem_centroids.dat");
		for(unsigned int i = 0 ; i < system.get_mesh().n_elem(); i++){
			Point elem_cent = system.get_mesh().elem(i)->centroid();
			if(output_elem_cent.is_open()){
				output_elem_cent << elem_cent(0) << " " << elem_cent(1) << "\n";
			}
		}
		output_elem_cent.close();
      
#ifdef LIBMESH_HAVE_EXODUS_API
    // Write out this timestep if we're requested to
      if ((t_step+1)%write_interval == 0)
	{
        std::ostringstream file_name;
	/*
        // We write the file in the ExodusII format.
        file_name << "out_"
                  << std::setw(3)
                  << std::setfill('0')
                  << std::right
                  << t_step+1
                  << ".e";
				//this should write out the primal which should be the same as what's read in...
				ExodusII_IO(mesh).write_timestep(file_name.str(),
								                        equation_systems,
								                        1, //number of time steps written to file
								                        system.time);
	*/
	}
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
    }
  
  // All done.
  return 0;
  
} //end main
