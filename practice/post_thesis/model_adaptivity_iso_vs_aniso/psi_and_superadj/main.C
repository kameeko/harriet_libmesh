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
#include "convdiff_sadj_primary.h"
#include "convdiff_sadj_aux.h"
#include "initial.h"

#include "libmesh/dof_map.h" //alternate error breakdown
#include "libmesh/direct_solution_transfer.h"

//FOR DEBUGGING
#include "libmesh/sparse_matrix.h" //DEBUG
#include "libmesh/gmv_io.h" //DEBUG

//bool greaterThan (std::pair<double,double> i, std::pair<double,double> j) { return (i.first > j.first); }

// The main program
int main(int argc, char** argv)
{
  clock_t begin = std::clock();
  
  // Initialize libMesh
  LibMeshInit init(argc, argv);
  
  // Parameters
  GetPot solverInfile("fem_system_params.in");
  const bool transient                  = solverInfile("transient", false);
  unsigned int n_timesteps              = solverInfile("n_timesteps", 1);
  const int nx                          = solverInfile("nx",100);
  const int ny                          = solverInfile("ny",100);
  const int nz                          = solverInfile("nz",100);
  GetPot infile("contamTrans.in");
  //std::string find_mesh_here            = infile("initial_mesh","mesh.exo");
  bool doContinuation                   = infile("do_continuation",false);
  bool splitSuperAdj                    = infile("split_super_adjoint",true); //solve as single adjoint or two forwards
  int maxIter                           = infile("max_model_refinements",0);  //maximum number of model refinements
  double refStep                        = infile("refinement_step",0.1); //additional proportion of domain refined per step
    //this refers to additional basis functions...number of elements will be more...
  double qoiErrorTol                    = infile("relative_error_tolerance",0.01); //stopping criterion
  //bool doDivvyMatlab                    = infile("do_divvy_in_Matlab",false); //output files to determine next refinement in Matlab

  if(refStep*maxIter > 1)
    maxIter = round(ceil(1./refStep));

  Mesh mesh(init.comm());
  Mesh mesh2(init.comm()); 
  
  //create mesh
  unsigned int dim;
  if(nz == 0){ //to check if oscillations happen in 2D as well...
    dim = 2;
    MeshTools::Generation::build_square(mesh, nx, ny, 0., 2300., 0., 1650., QUAD9);
    MeshTools::Generation::build_square(mesh2, nx, ny, 0., 2300., 0., 1650., QUAD9);
  }else{
    dim = 3;
    MeshTools::Generation::build_cube(mesh, nx, ny, nz, 0., 2300., 0., 1650., 0., 100., HEX27);
    MeshTools::Generation::build_cube(mesh2, nx, ny, nz, 0., 2300., 0., 1650., 0., 100., HEX27);
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
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
  equation_systems.init ();
  equation_systems_mix.init();
  
  //initial guess for primary state
  read_initial_parameters();
  system_primary.project_solution(initial_value, initial_grad,
                          equation_systems.parameters);
  finish_initialization();

  //nonlinear solver options
  NewtonSolver *solver_sadj_primary = new NewtonSolver(system_sadj_primary); 
  system_sadj_primary.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_sadj_primary); 
  solver_sadj_primary->quiet = solverInfile("solver_quiet", true);
  solver_sadj_primary->verbose = !solver_sadj_primary->quiet;
  solver_sadj_primary->max_nonlinear_iterations =
    solverInfile("max_nonlinear_iterations", 15);
  solver_sadj_primary->relative_step_tolerance =
    solverInfile("relative_step_tolerance", 1.e-3);
  solver_sadj_primary->relative_residual_tolerance =
    solverInfile("relative_residual_tolerance", 0.0);
  solver_sadj_primary->absolute_residual_tolerance =
    solverInfile("absolute_residual_tolerance", 0.0);
  NewtonSolver *solver_sadj_aux = new NewtonSolver(system_sadj_aux); 
  system_sadj_aux.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_sadj_aux); 
  solver_sadj_aux->quiet = solverInfile("solver_quiet", true);
  solver_sadj_aux->verbose = !solver_sadj_aux->quiet;
  solver_sadj_aux->max_nonlinear_iterations =
    solverInfile("max_nonlinear_iterations", 15);
  solver_sadj_aux->relative_step_tolerance =
    solverInfile("relative_step_tolerance", 1.e-3);
  solver_sadj_aux->relative_residual_tolerance =
    solverInfile("relative_residual_tolerance", 0.0);
  solver_sadj_aux->absolute_residual_tolerance =
    solverInfile("absolute_residual_tolerance", 0.0);
  NewtonSolver *solver_primary = new NewtonSolver(system_primary); 
  system_primary.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_primary); 
  solver_primary->quiet = solverInfile("solver_quiet", true);
  solver_primary->verbose = !solver_primary->quiet;
  solver_primary->max_nonlinear_iterations =
    solverInfile("max_nonlinear_iterations", 15);
  solver_primary->relative_step_tolerance =
    solverInfile("relative_step_tolerance", 1.e-3);
  solver_primary->relative_residual_tolerance =
    solverInfile("relative_residual_tolerance", 0.0);
  solver_primary->absolute_residual_tolerance =
    solverInfile("absolute_residual_tolerance", 0.0);
  NewtonSolver *solver_aux = new NewtonSolver(system_aux); 
  system_aux.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver_aux); 
  solver_aux->quiet = solverInfile("solver_quiet", true);
  solver_aux->verbose = !solver_aux->quiet;
  solver_aux->max_nonlinear_iterations =
    solverInfile("max_nonlinear_iterations", 15);
  solver_aux->relative_step_tolerance =
    solverInfile("relative_step_tolerance", 1.e-3);
  solver_aux->relative_residual_tolerance =
    solverInfile("relative_residual_tolerance", 0.0);
  solver_aux->absolute_residual_tolerance =
    solverInfile("absolute_residual_tolerance", 0.0);
    
  //linear solver options
  solver_primary->max_linear_iterations       = solverInfile("max_linear_iterations",10000);
  solver_primary->initial_linear_tolerance    = solverInfile("initial_linear_tolerance",1.e-13);
  solver_primary->minimum_linear_tolerance    = solverInfile("minimum_linear_tolerance",1.e-13);
  solver_primary->linear_tolerance_multiplier = solverInfile("linear_tolerance_multiplier",1.e-3);
  solver_aux->max_linear_iterations           = solverInfile("max_linear_iterations",10000);
  solver_aux->initial_linear_tolerance        = solverInfile("initial_linear_tolerance",1.e-13);
  solver_aux->minimum_linear_tolerance        = solverInfile("minimum_linear_tolerance",1.e-13);
  solver_aux->linear_tolerance_multiplier     = solverInfile("linear_tolerance_multiplier",1.e-3);
  solver_sadj_primary->max_linear_iterations       = solverInfile("max_linear_iterations",10000);
  solver_sadj_primary->initial_linear_tolerance    = solverInfile("initial_linear_tolerance",1.e-13);
  solver_sadj_primary->minimum_linear_tolerance    = solverInfile("minimum_linear_tolerance",1.e-13);
  solver_sadj_primary->linear_tolerance_multiplier = solverInfile("linear_tolerance_multiplier",1.e-3);
  solver_sadj_aux->max_linear_iterations           = solverInfile("max_linear_iterations",10000);
  solver_sadj_aux->initial_linear_tolerance        = solverInfile("initial_linear_tolerance",1.e-13);
  solver_sadj_aux->minimum_linear_tolerance        = solverInfile("minimum_linear_tolerance",1.e-13);
  solver_sadj_aux->linear_tolerance_multiplier     = solverInfile("linear_tolerance_multiplier",1.e-3);
  
  /*if(doDivvyMatlab){
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
    std::ofstream output_elem_cent("elem_centroids.dat");
    for(unsigned int i = 0 ; i < system_mix.get_mesh().n_elem(); i++){
      Point elem_cent = system_mix.get_mesh().elem(i)->centroid();
      if(output_elem_cent.is_open()){
        output_elem_cent << elem_cent(0) << " " << elem_cent(1) << "\n";
      }
    }
    output_elem_cent.close();
  }*/
  
  //inverse dof map (elements in support of each node, assuming every 6 dofs belong to same node)
  std::vector<std::set<dof_id_type> > node_to_elem; 
  node_to_elem.resize(round(system_mix.n_dofs()/6.));
  for(unsigned int i = 0 ; i < system_mix.get_mesh().n_elem(); i++){
    std::vector< dof_id_type > di;
    system_mix.get_dof_map().dof_indices(system_mix.get_mesh().elem(i), di);
    for(unsigned int j = 0; j < di.size(); j++)
      node_to_elem[round(floor(di[j]/6.))].insert(i);
  }

  int refIter = 0;
  double relError = 2.*qoiErrorTol;
  while(refIter <= maxIter && relError > qoiErrorTol){
    if(!doContinuation){ //clear out previous solutions
      system_primary.solution->zero();
      system_aux.solution->zero();
      system_sadj_primary.solution->zero();
      system_sadj_aux.solution->zero();
    }
    system_mix.solution->zero();
    
    clock_t begin_inv = std::clock();
    system_primary.solve();
    system_primary.clearQoI();
    clock_t end_inv = std::clock();
    clock_t begin_err_est = std::clock();
    std::cout << "\n End primary solve, begin auxiliary solve..." << std::endl;
    system_aux.solve();
    std::cout << "\n End auxiliary solve..." << std::endl;
    clock_t end_aux = std::clock();
    
    system_primary.postprocess();
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

    if(!splitSuperAdj){ //solve super-adjoint as single adjoint
      system_mix.assemble_qoi_sides = true; //QoI doesn't involve sides
      
      std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve start ~*~*~*~*~*~*~*~*~\n" << std::endl;
      std::pair<unsigned int, Real> adjsolve = system_mix.adjoint_solve();
      std::cout << "number of iterations to solve adjoint: " << adjsolve.first << std::endl;
      std::cout << "final residual of adjoint solve: " << adjsolve.second << std::endl;
      std::cout << "\n~*~*~*~*~*~*~*~*~ adjoint solve end ~*~*~*~*~*~*~*~*~" << std::endl;
  
      NumericVector<Number> &dual_sol = system_mix.get_adjoint_solution(0); //DEBUG
      
      system_mix.assemble(); //calculate residual to correspond to solution
      
      //std::cout << "\n sadj norm: " << system_mix.calculate_norm(dual_sol, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 0, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 1, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 2, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 3, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 4, L2) << std::endl; //DEBUG
      //std::cout << system_mix.calculate_norm(dual_sol, 5, L2) << std::endl; //DEBUG
    }else{ //solve super-adjoint as
      system_mix.assemble(); //calculate residual to correspond to solution
      std::cout << "\n Begin primary super-adjoint solve...\n" << std::endl;
      system_sadj_primary.solve();
      std::cout << "\n End primary super-adjoint solve, begin auxiliary super-adjoint solve...\n" << std::endl;
      system_sadj_aux.solve();
      std::cout << "\n End auxiliary super-adjoint solve...\n" << std::endl;
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
      //std::cout << "\n sadj norm: " << system_mix.calculate_norm(primal_sol, L2) << std::endl; //DEBUG
      dual_sol.swap(primal_sol);
      //std::cout << "\n sadj norm: " << system_mix.calculate_norm(primal_sol, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_primary.calculate_norm(*system_sadj_primary.solution, 0, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_primary.calculate_norm(*system_sadj_primary.solution, 1, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_primary.calculate_norm(*system_sadj_primary.solution, 2, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_aux.calculate_norm(*system_sadj_aux.solution, 0, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_aux.calculate_norm(*system_sadj_aux.solution, 1, L2) << std::endl; //DEBUG
      //std::cout << system_sadj_aux.calculate_norm(*system_sadj_aux.solution, 2, L2) << std::endl; //DEBUG
    }
    NumericVector<Number> &dual_solution = system_mix.get_adjoint_solution(0);
/*  
    NumericVector<Number> &primal_solution = *system_mix.solution; //DEBUG
    primal_solution.swap(dual_solution); //DEBUG
    ExodusII_IO(mesh).write_timestep("super_adjoint.exo",
                                   equation_systems,
                                   1, /  his number indicates how many time steps are being written to the file
                                   system_mix.time); //DEBUG
    primal_solution.swap(dual_solution); //DEBUG
*/
    //adjoint-weighted residual
    AutoPtr<NumericVector<Number> > adjresid = system_mix.solution->zero_clone();
    adjresid->pointwise_mult(*system_mix.rhs,dual_solution); 
    adjresid->scale(-0.5);
    std::cout << "\n -0.5*M'_HF(psiLF)(superadj): " << adjresid->sum() << std::endl; //DEBUG
    
    //LprimeHF(psiLF)
    AutoPtr<NumericVector<Number> > LprimeHF_psiLF = system_mix.solution->zero_clone();
    LprimeHF_psiLF->pointwise_mult(*system_mix.rhs,*just_aux);
    std::cout << " L'_HF(psiLF): " << LprimeHF_psiLF->sum() << std::endl; //DEBUG
    
    //QoI and error estimate
    std::cout << "QoI: " << std::setprecision(17) << system_primary.getQoI() << std::endl;
    std::cout << "QoI Error estimate: " << std::setprecision(17) 
        << adjresid->sum()+LprimeHF_psiLF->sum() << std::endl; 
        
    relError = fabs((adjresid->sum()+LprimeHF_psiLF->sum())/system_primary.getQoI());
    
    //print out information
    std::cout << "Estimated absolute relative qoi error: " << relError << std::endl << std::endl;
    std::cout << "Estimated HF QoI: " << std::setprecision(17) << system_primary.getQoI()+adjresid->sum()+LprimeHF_psiLF->sum() << std::endl;
    clock_t end = clock();
    std::cout << "Time so far: " << double(end-begin)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "Time for inverse problem: " << double(end_inv-begin_inv)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "Time for extra error estimate bits: " << double(end-begin_err_est)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "    Time to get auxiliary problems: " << double(end_aux-begin_err_est)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "    Time to get superadjoint: " << double(end-end_aux)/CLOCKS_PER_SEC << " seconds...\n" << std::endl;
    
    //proportion of domain refined at this iteration
    MeshBase::element_iterator       elem_it  = mesh.elements_begin();
    const MeshBase::element_iterator elem_end = mesh.elements_end();
    double numMarked = 0.;
    for (; elem_it != elem_end; ++elem_it){
      Elem* elem = *elem_it;
      numMarked += elem->subdomain_id();
    }
    std::cout << "Refinement fraction: " << numMarked/system_mix.get_mesh().n_elem() << std::endl << std::endl;

    
    //output at each iteration
    std::stringstream ss;
    ss << refIter;
    std::string str = ss.str();
    std::string write_error_basis_blame = 
      (infile("error_est_output_file_basis_blame", "error_est_breakdown_basis_blame")) + str + ".dat";
    std::ofstream output2(write_error_basis_blame);
    for(unsigned int i = 0 ; i < adjresid->size(); i++){
      if(output2.is_open())
        output2 << (*adjresid)(i) + (*LprimeHF_psiLF)(i) << "\n";
    }
    output2.close();
    
    if(refIter < maxIter && relError > qoiErrorTol){ //if further refinement needed
    
      //collapse error contributions into nodes
      std::vector<std::pair<Number,dof_id_type> > node_errs(round(system_mix.n_dofs()/6.));
      for(unsigned int node_num = 0; node_num < node_errs.size(); node_num++){
        node_errs[node_num] = std::pair<Number,dof_id_type>
                             (fabs((*adjresid)(6*node_num) + (*LprimeHF_psiLF)(6*node_num)
                            + (*adjresid)(6*node_num+1) + (*LprimeHF_psiLF)(6*node_num+1)
                            + (*adjresid)(6*node_num+2) + (*LprimeHF_psiLF)(6*node_num+2)
                            + (*adjresid)(6*node_num+3) + (*LprimeHF_psiLF)(6*node_num+3)
                            + (*adjresid)(6*node_num+4) + (*LprimeHF_psiLF)(6*node_num+4)
                            + (*adjresid)(6*node_num+5) + (*LprimeHF_psiLF)(6*node_num+5)), node_num);
      }
  
      //find nodes contributing the most
      //double refPcnt = std::min((refIter+1)*refStep,1.);
      double refPcnt = std::min(refStep,1.); //additional refinement (compared to previous iteration)
      int cutoffLoc = round(node_errs.size()*refPcnt);
      std::sort(node_errs.begin(), node_errs.end()); 
      std::reverse(node_errs.begin(), node_errs.end()); 
      
      //find elements in support of worst offenders
      std::vector<dof_id_type> markMe;
      markMe.reserve(cutoffLoc*8);
      for(int i = 0; i < cutoffLoc; i++){
        markMe.insert(markMe.end(), node_to_elem[node_errs[i].second].begin(), node_to_elem[node_errs[i].second].end());
      }
  
      //mark those elements for refinement.
      for(int i = 0; i < markMe.size(); i++){
        mesh.elem(markMe[i])->subdomain_id() = 1; //assuming HF regions marked with 1
      }
  
      //to test whether assignment matches matlab's
      /*std::string stash_assign = "divvy_c_poke.txt";
      std::ofstream output_dbg(stash_assign.c_str());
      MeshBase::element_iterator       elem_it  = mesh.elements_begin();
      const MeshBase::element_iterator elem_end = mesh.elements_end();
      for (; elem_it != elem_end; ++elem_it){
        Elem* elem = *elem_it;
            
        if(output_dbg.is_open()){
          output_dbg << elem->id() << " " << elem->subdomain_id() << "\n";
        }
      }
      output_dbg.close();*/
      
#ifdef LIBMESH_HAVE_EXODUS_API
    std::stringstream ss2;
    ss2 << refIter + 1;
    std::string str = ss2.str();
    std::string write_divvy = "divvy" + str + ".exo";
    ExodusII_IO (mesh).write_equation_systems(write_divvy,equation_systems); //DEBUG
#endif // #ifdef LIBMESH_HAVE_EXODUS_API
    }
    refIter += 1;
  }
  
  std::string stash_assign = "divvy_final.txt";
  std::ofstream output_dbg(stash_assign.c_str());
  MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  double numMarked = 0.;
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    numMarked += elem->subdomain_id();
    if(output_dbg.is_open()){
      output_dbg << elem->id() << " " << elem->subdomain_id() << "\n";
    }
  }
  output_dbg.close();
  
  std::cout << "\nRefinement concluded..." << std::endl;
  std::cout << "Final refinement fraction: " << numMarked/system_mix.get_mesh().n_elem() << std::endl;
  std::cout << "Final estimated relative error: " << relError << std::endl;
  
  return 0; //done

} //end main
