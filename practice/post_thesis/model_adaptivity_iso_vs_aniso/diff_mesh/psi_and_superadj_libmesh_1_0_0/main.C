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
#include "libmesh/mesh_function.h"

//FOR DEBUGGING
#include "libmesh/sparse_matrix.h" //DEBUG
#include "libmesh/gmv_io.h" //DEBUG

//#ifdef LIBMESH_TRILINOS_HAVE_DTK
//#include "libmesh/dtk_solution_transfer.h"
//#endif

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
  const int nx_LF                       = solverInfile("nx_LF",100);
  const int ny_LF                       = solverInfile("ny_LF",100);
  const int nz_LF                       = solverInfile("nz_LF",100);
  const int nx_ratio                    = solverInfile("nx_ratio",2);
  const int ny_ratio                    = solverInfile("ny_ratio",2);
  const int nz_ratio                    = solverInfile("nz_ratio",2);
  const bool xfer_if_possible           = solverInfile("direct_transfer_if_possible",true); //DEBUG
  GetPot infile("contamTrans.in");
  //std::string find_mesh_here            = infile("initial_mesh","mesh.exo");
  bool doContinuation                   = infile("do_continuation",false);
  int maxIter                           = infile("max_model_refinements",0);  //maximum number of model refinements
  double refStep                        = infile("refinement_step",0.1); //additional proportion of domain refined per step
    //this refers to additional basis functions...number of elements will be more...
  double qoiErrorTol                    = infile("relative_error_tolerance",0.01); //stopping criterion
  //bool doDivvyMatlab                    = infile("do_divvy_in_Matlab",false); //output files to determine next refinement in Matlab

  if(refStep*maxIter > 1)
    maxIter = round(ceil(1./refStep));

  Mesh mesh(init.comm()); //low/mixed-fidelity mesh
  Mesh mesh_HF(init.comm()); //high-fidelity mesh, for super-adj
  //Mesh mesh_HF_dbg(init.comm());
  
  int n_LF_elems = mesh.n_elem();
  
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.max_h_level() = 1;
  
  //create mesh
  unsigned int dim;
  const int nx_HF = nx_ratio*nx_LF;
  const int ny_HF = ny_ratio*ny_LF;
  const int nz_HF = nz_ratio*nz_LF;
  const double Lx = 2300; //4600
  const double Ly = 1650; //3300
  const double Lz = 100;
  if((nx_ratio != ny_ratio) || (nx_ratio != nz_ratio) || (ny_ratio != nz_ratio))
    std::cout << "\n\nAAAAAHHHHH can't refine element in only some of its dimensions...\n\n" << std::endl;
  if(nz_LF == 0){ //to check if oscillations happen in 2D as well...
    dim = 2;
    MeshTools::Generation::build_square(mesh, nx_LF, ny_LF, 0., Lx, 0., Ly, QUAD9);
    MeshTools::Generation::build_square(mesh_HF, nx_HF, ny_HF, 0., Lx, 0., Ly, QUAD9);
    //MeshTools::Generation::build_square(mesh_LF, nx_LF, ny_LF, 0., Lx, 0., Ly, QUAD9);
  }else{
    dim = 3;
    MeshTools::Generation::build_cube(mesh, nx_LF, ny_LF, nz_LF, 0., Lx, 0., Ly, 0., Lz, HEX27);
    MeshTools::Generation::build_cube(mesh_HF, nx_HF, ny_HF, nz_HF, 0., Lx, 0., Ly, 0., Lz, HEX27);
    //MeshTools::Generation::build_cube(mesh_LF, nx_LF, ny_LF, nz_LF, 0., Lx, 0., Ly, 0., Lz, HEX27);
    //MeshTools::Generation::build_cube(mesh_HF_dbg, nx_HF, ny_HF, nz_HF, 0., Lx, 0., Ly, 0., Lz, HEX27); //DEBUG
  }
  double dx = Lx/nx_HF;
  double dy = Ly/ny_HF;
  double dz = 0.;
  if(dim == 3){dz = Lz/nz_HF; }
  
  mesh.print_info(); //DEBUG
  mesh_HF.print_info(); //DEBUG
  
  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  EquationSystems equation_systems_mix(mesh_HF);
  //EquationSystems equation_systems_dbg(mesh_HF_dbg); //DEBUG

  //systems - coarser mesh
  ConvDiff_PrimarySys & system_primary = 
    equation_systems.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys"); //for primary variables
  ConvDiff_AuxSys & system_aux = 
    equation_systems.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys"); //for auxiliary variables
  //ConvDiff_MprimeSys & system_mix_MF = 
  //  equation_systems.add_system<ConvDiff_MprimeSys>("Diff_ConvDiff_MprimeSys"); //for psi
  
  //DEBUG
  //ConvDiff_PrimarySys & system_primary_dbg = 
  //  equation_systems_dbg.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys"); //for primary variables
  //ConvDiff_AuxSys & system_aux_dbg = 
  //  equation_systems_dbg.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys"); //for auxiliary variables
    
  //systems - fine mesh
  ConvDiff_MprimeSys & system_mix = 
    equation_systems_mix.add_system<ConvDiff_MprimeSys>("Diff_ConvDiff_MprimeSys"); //for psi and superadj
  ConvDiff_PrimarySys & system_primary_proj = 
    equation_systems_mix.add_system<ConvDiff_PrimarySys>("ConvDiff_PrimarySys"); //for primary variables
  ConvDiff_AuxSys & system_aux_proj = 
    equation_systems_mix.add_system<ConvDiff_AuxSys>("ConvDiff_AuxSys"); //for auxiliary variables
  ConvDiff_PrimarySadjSys & system_sadj_primary = 
    equation_systems_mix.add_system<ConvDiff_PrimarySadjSys>("ConvDiff_PrimarySadjSys"); //for split superadj
  ConvDiff_AuxSadjSys & system_sadj_aux = 
    equation_systems_mix.add_system<ConvDiff_AuxSadjSys>("ConvDiff_AuxSadjSys"); //for split superadj
    
  //keep solution at previous refinement iteration as init guess
  system_primary.project_solution_on_reinit() = true;
  system_aux.project_solution_on_reinit() = true;
  //system_mix_MF.project_solution_on_reinit() = true;
  system_primary_proj.project_solution_on_reinit() = true;
  system_aux_proj.project_solution_on_reinit() = true;
  
  //steady-state problem  
  system_primary.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_primary));
  system_aux.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_aux));
  //system_mix_MF.time_solver =
  //  UniquePtr<TimeSolver>(new SteadySolver(system_mix_MF));
  system_primary_proj.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_primary_proj));
  system_aux_proj.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_aux_proj));
  system_mix.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_mix));
  system_sadj_primary.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_sadj_primary));
  system_sadj_aux.time_solver =
    UniquePtr<TimeSolver>(new SteadySolver(system_sadj_aux));
  //system_primary_dbg.time_solver =
  //  UniquePtr<TimeSolver>(new SteadySolver(system_primary_dbg)); //DEBUG
  //system_aux_dbg.time_solver =
  //  UniquePtr<TimeSolver>(new SteadySolver(system_aux_dbg)); //DEBUG
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
  equation_systems.init();
  equation_systems_mix.init();
  //equation_systems_dbg.init();
  
  //initial guess for primary state
  read_initial_parameters();
  system_primary.project_solution(initial_value, initial_grad,
                          equation_systems.parameters);
  finish_initialization();

  //nonlinear solver options
  NewtonSolver *solver_sadj_primary = new NewtonSolver(system_sadj_primary); 
  system_sadj_primary.time_solver->diff_solver() = UniquePtr<DiffSolver>(solver_sadj_primary); 
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
  solver_sadj_primary->require_residual_reduction = solverInfile("require_residual_reduction",true);
  NewtonSolver *solver_sadj_aux = new NewtonSolver(system_sadj_aux); 
  system_sadj_aux.time_solver->diff_solver() = UniquePtr<DiffSolver>(solver_sadj_aux); 
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
  solver_sadj_aux->require_residual_reduction = solverInfile("require_residual_reduction",true);
  NewtonSolver *solver_primary = new NewtonSolver(system_primary); 
  system_primary.time_solver->diff_solver() = UniquePtr<DiffSolver>(solver_primary); 
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
  solver_primary->require_residual_reduction = solverInfile("require_residual_reduction",true);
  NewtonSolver *solver_aux = new NewtonSolver(system_aux); 
  system_aux.time_solver->diff_solver() = UniquePtr<DiffSolver>(solver_aux); 
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
  solver_aux->require_residual_reduction = solverInfile("require_residual_reduction",true);
  
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
  
  //inverse dof map for LF mesh (elements in support of each node, assuming every nvar dofs belong to same node)
  std::vector<std::set<dof_id_type> > node_to_elem; 
  node_to_elem.resize(round(system_aux.n_dofs()/3.));
  for(unsigned int i = 0 ; i < system_aux.get_mesh().n_elem(); i++){
    std::vector< dof_id_type > di;
    system_aux.get_dof_map().dof_indices(system_aux.get_mesh().elem(i), di);
    for(unsigned int j = 0; j < di.size(); j++)
      node_to_elem[round(floor(di[j]/3.))].insert(i);
  }
  
  bool mesh_diff = false; //whether two models should differ in mesh
  if(nx_ratio > 1 || ny_ratio > 1 || nz_ratio > 1)
    mesh_diff = true;
  
  //find fine elements contained by coarse elements
  int elem_ratio = nx_ratio*ny_ratio;
  if(dim > 2)
    elem_ratio *= nz_ratio;
  MeshBase::element_iterator       elem_it_LF  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end_LF = mesh.elements_end();
  MeshBase::element_iterator       elem_it_HF  = mesh_HF.elements_begin();
  const MeshBase::element_iterator elem_end_HF = mesh_HF.elements_end();
  std::vector<std::set<dof_id_type> > elem_mapping;
  if(mesh_diff){
    elem_mapping.resize(mesh.n_elem());
    int n_babies_found = 0;
    for (; elem_it_LF != elem_end_LF; ++elem_it_LF){
      Elem* elem = *elem_it_LF;
      Point elem_cent = elem->centroid();
      n_babies_found = 0;
      elem_it_HF = mesh_HF.elements_begin();
      for (; elem_it_HF != elem_end_HF; ++elem_it_HF){
        if(n_babies_found >= elem_ratio)
          break;
        Elem* elem_HF = *elem_it_HF;
        Point elem_cent_diff = elem_HF->centroid();
        elem_cent_diff.subtract(elem_cent);
        if(std::abs(elem_cent_diff(0)) < 0.51*dx*(nx_ratio-1) && 
            std::abs(elem_cent_diff(1)) < 0.51*dy*(ny_ratio-1) && 
            std::abs(elem_cent_diff(2)) < 0.51*dz*(nz_ratio-1)){
          elem_mapping[elem->id()].insert(elem_HF->id());
          n_babies_found += 1;   
        }
      }
    }
  }else{ //this mapping not needed
    elem_mapping.resize(0);
  }
  
  //mapping fine basis functions to coarse basis functions that they overlap
  std::vector<std::set<dof_id_type> > fine_to_coarse_nodes;
  if(mesh_diff){
    fine_to_coarse_nodes.resize(round(system_mix.n_dofs()/3.));
    elem_it_LF = mesh.elements_begin();
    for (; elem_it_LF != elem_end_LF; ++elem_it_LF){ //for each coarse element
      Elem* elem = *elem_it_LF;
      std::vector< dof_id_type > di_LF;
      system_aux.get_dof_map().dof_indices(elem, di_LF);
      for (dof_id_type baby_id : elem_mapping[elem->id()]){ //for each fine element in the coarse element
        std::vector< dof_id_type > di_HF;
        system_sadj_primary.get_dof_map().dof_indices(system_sadj_primary.get_mesh().elem(baby_id), di_HF);
        for(unsigned int j = 0; j < di_LF.size(); j++){ //for each coarse node in elem
          dof_id_type node_LF = round(floor(di_LF[j]/3.));
          for(unsigned int k = 0; k < di_LF.size(); k++){ //for each fine node in elem
            dof_id_type node_HF = round(floor(di_HF[k]/3.));
            fine_to_coarse_nodes[node_HF].insert(node_LF);
          }
        }
      }
    }
  }else{ //this mapping not needed
    fine_to_coarse_nodes.resize(0);
  }
  
  int refIter = 0;
  double relError = 2.*qoiErrorTol;
  while(refIter <= maxIter && relError > qoiErrorTol){
    //equation_systems.reinit(); //project previous solution onto new mesh
    //equation_systems_mix_MF.reinit(); 
    
    if(!doContinuation){ //clear out previous solutions
      system_primary.solution->zero();
      system_aux.solution->zero();
      system_sadj_primary.solution->zero();
      system_sadj_aux.solution->zero();
    }
    system_mix.solution->zero();
    //system_mix_MF.solution->zero();
    
    std::cout << "\n Begin primary solve..." << std::endl;
    clock_t begin_inv = std::clock();
    system_primary.solve();
    system_primary.clearQoI();
    clock_t end_inv = std::clock();
    clock_t begin_err_est = std::clock();
    std::cout << "\n End primary solve, begin auxiliary solve..." << std::endl;
    system_aux.solve();
    std::cout << "\n End auxiliary solve..." << std::endl;
    
    system_primary.postprocess();
    system_aux.postprocess();
/*
std::ostringstream Jfile_name1;
Jfile_name1 << "J_primary.dat";
std::ofstream outputJ1(Jfile_name1.str());
system_primary.matrix->print(outputJ1);
outputJ1.close();
std::ostringstream Jfile_name2;
Jfile_name2 << "J_aux.dat";
std::ofstream outputJ2(Jfile_name2.str());
system_aux.matrix->print(outputJ2);
outputJ2.close();
*/    
    system_sadj_primary.set_c_vals(system_primary.get_c_vals());
    system_sadj_aux.set_auxc_vals(system_aux.get_auxc_vals());

    std::cout << "\nQoI: " << std::setprecision(17) << system_primary.getQoI() << std::endl;

    equation_systems_mix.reinit();
    
    //bug (?) with FEMContext::interior_values means we can't transfer over psi as a whole
/*    
    //combine primary and auxiliary variables into psi
    DirectSolutionTransfer sol_transfer(init.comm()); 
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_c")),
      system_mix_MF.variable(system_mix_MF.variable_number("aux_c")));
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_zc")),
      system_mix_MF.variable(system_mix_MF.variable_number("aux_zc")));
    sol_transfer.transfer(system_aux.variable(system_aux.variable_number("aux_fc")),
      system_mix_MF.variable(system_mix_MF.variable_number("aux_fc")));
    std::vector<dof_id_type> all_the_vars;
    system_mix_MF.get_all_variable_numbers(all_the_vars);
    MeshFunction* psi_MF_meshfx = new libMesh::MeshFunction(equation_systems, *system_mix_MF.solution, system_mix_MF.get_dof_map(), all_the_vars);
    psi_MF_meshfx->init();
    system_mix.project_solution(psi_MF_meshfx); 

    UniquePtr<NumericVector<Number> > just_aux = system_mix.solution->clone(); //project just aux vars first
    
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("c")),
      system_mix_MF.variable(system_mix_MF.variable_number("c")));
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("zc")),
      system_mix_MF.variable(system_mix_MF.variable_number("zc")));
    sol_transfer.transfer(system_primary.variable(system_primary.variable_number("fc")),
      system_mix_MF.variable(system_mix_MF.variable_number("fc")));
    system_mix.project_solution(psi_MF_meshfx); //project all of psi
*/     

#ifdef LIBMESH_HAVE_EXODUS_API
    //ExodusII_IO (mesh).write_equation_systems("pre_proj.exo",equation_systems); //DEBUG
#endif // #ifdef LIBMESH_HAVE_EXODUS_API 
        
    //project variables
    std::cout << "Begin projecting psi...\n" << std::endl;
    clock_t begin_proj = std::clock();
    if(mesh_diff || !xfer_if_possible){
      std::cout << "...actually projecting..." << std::endl;
      std::vector<dof_id_type> primary_vars;
      std::vector<dof_id_type> aux_vars;
      //system_primary.get_all_variable_numbers(primary_vars); //out of order, and order matters
      //system_aux.get_all_variable_numbers(aux_vars); //out of order, and order matters...
      primary_vars.push_back(0); primary_vars.push_back(1); primary_vars.push_back(2); 
      aux_vars.push_back(0); aux_vars.push_back(1); aux_vars.push_back(2); 
      MeshFunction* primary_MF_meshfx = 
        new libMesh::MeshFunction(equation_systems, 
                                  *system_primary.solution, 
                                  system_primary.get_dof_map(), 
                                  primary_vars);
      MeshFunction* aux_MF_meshfx = 
        new libMesh::MeshFunction(equation_systems, 
                                  *system_aux.solution, 
                                  system_aux.get_dof_map(), 
                                  aux_vars);
      primary_MF_meshfx->init();
      aux_MF_meshfx->init();
      system_primary_proj.project_solution(primary_MF_meshfx);
      system_aux_proj.project_solution(aux_MF_meshfx);

      //DEBUG
      //system_primary_dbg.project_solution(primary_MF_meshfx);
      //system_aux_dbg.project_solution(aux_MF_meshfx);

#ifdef LIBMESH_HAVE_EXODUS_API
    //ExodusII_IO (mesh_HF_dbg).write_equation_systems("post_proj.exo",equation_systems_dbg); //DEBUG
#endif // #ifdef LIBMESH_HAVE_EXODUS_API     

      delete primary_MF_meshfx; //avoid memory leakage, not sure why UniquePtr didn't help...
      delete aux_MF_meshfx; //avoid memory leakage, not sure why UniquePtr didn't help...

    }else{
      std::cout << "...actually doing direct transfer..." << std::endl;
      DirectSolutionTransfer sol_xfer(init.comm());
      sol_xfer.transfer(system_aux.variable(system_aux.variable_number("aux_c")),
        system_aux_proj.variable(system_aux_proj.variable_number("aux_c")));
      sol_xfer.transfer(system_aux.variable(system_aux.variable_number("aux_zc")),
        system_aux_proj.variable(system_aux_proj.variable_number("aux_zc")));
      sol_xfer.transfer(system_aux.variable(system_aux.variable_number("aux_fc")),
        system_aux_proj.variable(system_aux_proj.variable_number("aux_fc")));
      sol_xfer.transfer(system_primary.variable(system_primary.variable_number("c")),
        system_primary_proj.variable(system_primary_proj.variable_number("c")));
      sol_xfer.transfer(system_primary.variable(system_primary.variable_number("zc")),
        system_primary_proj.variable(system_primary_proj.variable_number("zc")));
      sol_xfer.transfer(system_primary.variable(system_primary.variable_number("fc")),
        system_primary_proj.variable(system_primary_proj.variable_number("fc")));
    } 
    std::cout << "\nFinished projecting psi...\n" << std::endl;
    clock_t end_proj = std::clock();

    //DEBUG
    //std::cout << system_primary_proj.calculate_norm(*system_primary_proj.solution, 0, L2) << std::endl; //DEBUG 
    //std::cout << system_primary_proj.calculate_norm(*system_primary_proj.solution, 1, L2) << std::endl; //DEBUG
    //std::cout << system_primary_proj.calculate_norm(*system_primary_proj.solution, 2, L2) << std::endl; //DEBUG
    //std::cout << system_aux_proj.calculate_norm(*system_aux_proj.solution, 0, L2) << std::endl; //DEBUG
    //std::cout << system_aux_proj.calculate_norm(*system_aux_proj.solution, 1, L2) << std::endl; //DEBUG
    //std::cout << system_aux_proj.calculate_norm(*system_aux_proj.solution, 2, L2) << std::endl; //DEBUG

    //combine into one psi
    DirectSolutionTransfer sol_transfer(init.comm()); 
    sol_transfer.transfer(system_aux_proj.variable(system_aux_proj.variable_number("aux_c")),
      system_mix.variable(system_mix.variable_number("aux_c")));
    sol_transfer.transfer(system_aux_proj.variable(system_aux_proj.variable_number("aux_zc")),
      system_mix.variable(system_mix.variable_number("aux_zc")));
    sol_transfer.transfer(system_aux_proj.variable(system_aux_proj.variable_number("aux_fc")),
      system_mix.variable(system_mix.variable_number("aux_fc")));
    UniquePtr<NumericVector<Number> > just_aux = system_mix.solution->clone(); //project just aux vars first
    sol_transfer.transfer(system_primary_proj.variable(system_primary_proj.variable_number("c")),
      system_mix.variable(system_mix.variable_number("c")));
    sol_transfer.transfer(system_primary_proj.variable(system_primary_proj.variable_number("zc")),
      system_mix.variable(system_mix.variable_number("zc")));
    sol_transfer.transfer(system_primary_proj.variable(system_primary_proj.variable_number("fc")),
      system_mix.variable(system_mix.variable_number("fc")));

    system_mix.assemble(); //calculate residual to correspond to solution
    
    //super adjoint solve
    std::cout << "\n Begin primary super-adjoint solve...\n" << std::endl;
    system_sadj_primary.solve();
    std::cout << "\n End primary super-adjoint solve, being auxiliary super-adjoint solve...\n" << std::endl;
/*
std::ostringstream Jfile_name;
Jfile_name << "J_sadj_primary.dat";
std::ofstream outputJ(Jfile_name.str());
system_sadj_primary.matrix->print(outputJ);
outputJ.close();
*/
    system_sadj_aux.solve();
    std::cout << "\n End super-adjoint solves...\n" << std::endl;
    
    //combine super adjoint parts into one
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
    UniquePtr<NumericVector<Number> > adjresid = system_mix.solution->zero_clone();
    adjresid->pointwise_mult(*system_mix.rhs,dual_solution); 
    adjresid->scale(-0.5);
    std::cout << "\n -0.5*M'_HF(psiLF)(superadj): " << adjresid->sum() << std::endl; //DEBUG
    
    //LprimeHF(psiLF)
    UniquePtr<NumericVector<Number> > LprimeHF_psiLF = system_mix.solution->zero_clone();
    LprimeHF_psiLF->pointwise_mult(*system_mix.rhs,*just_aux);
    std::cout << " L'_HF(psiLF): " << LprimeHF_psiLF->sum() << std::endl; //DEBUG
    
    //QoI and error estimate
    //std::cout << "QoI: " << std::setprecision(17) << system_primary.getQoI() << std::endl;
    std::cout << "QoI Error estimate: " << std::setprecision(17) 
        << adjresid->sum()+LprimeHF_psiLF->sum() << std::endl; 
        
    relError = fabs((adjresid->sum()+LprimeHF_psiLF->sum())/system_primary.getQoI());
    std::cout << "Estimated relative qoi error: " << relError << std::endl;
    
    //informative outputs
    clock_t end = clock();
    std::cout << "Time so far: " << double(end-begin)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "Time for inverse problem: " << double(end_inv-begin_inv)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "Time to project psi: " << double(end_proj-begin_proj)/CLOCKS_PER_SEC << " seconds..." << std::endl;
    std::cout << "Time for extra error estimate bits: " << double(end-begin_err_est)/CLOCKS_PER_SEC << " seconds...\n" << std::endl;
    MeshBase::element_iterator       elem_it  = mesh.elements_begin();
    const MeshBase::element_iterator elem_end = mesh.elements_end();
    double numMarked = 0.;
    for (; elem_it != elem_end; ++elem_it){
      Elem* elem = *elem_it;
      numMarked += elem->subdomain_id(); //assumes HF is subdomain 1, LF is subdomain 0
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

      double dbg_sum1 = 0.; //DEBUG
      double dbg_sum2 = 0.; //DEBUG

      //collapse error contributions into nodes
      std::vector<std::pair<Number,dof_id_type> > node_errs(round(system_mix.n_dofs()/6.));
      for(unsigned int node_num = 0; node_num < node_errs.size(); node_num++){
        node_errs[node_num] = std::pair<Number,dof_id_type>
                             ((*adjresid)(6*node_num) + (*LprimeHF_psiLF)(6*node_num)
                            + (*adjresid)(6*node_num+1) + (*LprimeHF_psiLF)(6*node_num+1)
                            + (*adjresid)(6*node_num+2) + (*LprimeHF_psiLF)(6*node_num+2)
                            + (*adjresid)(6*node_num+3) + (*LprimeHF_psiLF)(6*node_num+3)
                            + (*adjresid)(6*node_num+4) + (*LprimeHF_psiLF)(6*node_num+4)
                            + (*adjresid)(6*node_num+5) + (*LprimeHF_psiLF)(6*node_num+5), node_num);
        dbg_sum1 += node_errs[node_num].first; //DEBUG
      }
      std::cout << "Should match QoI error estimate: " << dbg_sum1 << std::endl; //DEBUG
      
      //redistributed error contributions from fine to coarse nodes
      std::vector<std::pair<Number,dof_id_type> > node_errs_coarse(round(system_primary.n_dofs()/3.));
      if(mesh_diff){
        for(unsigned int node_num = 0; node_num < node_errs_coarse.size(); node_num++){ //initialize pairs
          node_errs_coarse[node_num] = std::pair<Number,dof_id_type>(0.0, node_num);
        }
        for(unsigned int fine_node = 0; fine_node < fine_to_coarse_nodes.size(); fine_node++){ //redistribute error
          for(dof_id_type coarse_node : fine_to_coarse_nodes[fine_node]){
            node_errs_coarse[coarse_node].first += node_errs[fine_node].first/fine_to_coarse_nodes[fine_node].size();
          }
        }
        for(unsigned int node_num = 0; node_num < node_errs_coarse.size(); node_num++){ //magnitudes of redistributed errors
          Number meep = node_errs_coarse[node_num].first;
          dbg_sum2 += meep; //DEBUG
          node_errs_coarse[node_num] = std::pair<Number,dof_id_type>(fabs(meep), node_num);
        }
      }else{
        for(unsigned int node_num = 0; node_num < node_errs_coarse.size(); node_num++){
          node_errs_coarse[node_num] = node_errs[node_num];
          dbg_sum2 += node_errs_coarse[node_num].first; //DEBUG
        }
      }
      std::cout << "Should match QoI error estimate: " << dbg_sum2 << std::endl; //DEBUG
      
      //find nodes contributing the most
      //double refPcnt = std::min((refIter+1)*refStep,1.);
      double refPcnt = std::min(refStep,1.); //additional refinement (compared to previous iteration)
      int cutoffLoc = round(node_errs_coarse.size()*refPcnt);
      std::sort(node_errs_coarse.begin(), node_errs_coarse.end()); 
      std::reverse(node_errs_coarse.begin(), node_errs_coarse.end()); 
      
      //find elements in support of worst offenders
      std::vector<dof_id_type> markMe;
      if(dim == 2)
        markMe.reserve(cutoffLoc*4);
      else if(dim == 3)
        markMe.reserve(cutoffLoc*8);
      for(int i = 0; i < cutoffLoc; i++){
        markMe.insert(markMe.end(), node_to_elem[node_errs_coarse[i].second].begin(), node_to_elem[node_errs_coarse[i].second].end());
      } 
      
      if(mesh_diff){
        //mark those elements for refinement
        mesh_refinement.clean_refinement_flags(); //remove all refinement flags
        for(int i = 0; i < markMe.size(); i++){
          if(mesh.elem(markMe[i])->active()) //don't mark already-refined elements
            mesh.elem(markMe[i])->set_refinement_flag(Elem::REFINE);
        }
        
        int prev_nelems = mesh.n_elem();
        mesh_refinement.refine_elements(); //refine to new MF mesh ...dies here at second refinement??
          //mesh_test is fine with asking to marking the same element for refinement multiple times...even compiled against new libmesh
        int new_nelems = mesh.n_elem();
        
        //mark new elements as HF subdomain = 1
        for(int i = 0; i < (new_nelems-prev_nelems); i++){
          mesh.elem(prev_nelems+i)->subdomain_id() = 1;
        }
      }else{
        for(int i = 0; i < markMe.size(); i++){
            mesh.elem(markMe[i])->subdomain_id() = 1;
        }
      }

      equation_systems.reinit(); //project previous solution onto new mesh
      
#ifdef LIBMESH_HAVE_EXODUS_API
    std::stringstream ss2;
    ss2 << refIter + 1;
    std::string str = ss2.str();
    std::string write_divvy = "divvy" + str + ".exo";
    ExodusII_IO (mesh).write_equation_systems(write_divvy,equation_systems); 
#endif // #ifdef LIBMESH_HAVE_EXODUS_API

    } //end refinement loop

    refIter += 1;
  }
  
  std::string stash_assign = "divvy_final.txt";
  std::ofstream output_dbg(stash_assign.c_str());
  MeshBase::element_iterator       elem_it  = mesh.elements_begin();
  const MeshBase::element_iterator elem_end = mesh.elements_end();
  double numMarked = 0.;
  for (; elem_it != elem_end; ++elem_it){
    Elem* elem = *elem_it;
    numMarked += elem->subdomain_id(); //assumes HF is subdomain 1, LF is subdomain 0
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
