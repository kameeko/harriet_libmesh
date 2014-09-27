// C++ includes
#include <iostream>
#include <iomanip>

// General libMesh includes
#include "libmesh/equation_systems.h"
#include "libmesh/linear_solver.h"
#include "libmesh/error_vector.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/newton_solver.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/steady_solver.h"
#include "libmesh/system_norm.h"

// Error Estimator includes
#include "libmesh/kelly_error_estimator.h"
#include "libmesh/patch_recovery_error_estimator.h"

// Adjoint Related includes
#include "libmesh/adjoint_residual_error_estimator.h"
#include "libmesh/qoi_set.h"

// Sensitivity Calculation related includes
#include "libmesh/parameter_vector.h"
#include "libmesh/sensitivity_data.h"

// libMesh I/O includes
#include "libmesh/getpot.h"
#include "libmesh/gmv_io.h"

// Local includes
#include "femparameters.h"
#include "convdiffstokes_sys.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

void write_output(EquationSystems &es,
                  unsigned int a_step,       // The adaptive step count
                  std::string solution_type) // primal or adjoint solve
{
#ifdef LIBMESH_HAVE_GMV
  MeshBase &mesh = es.get_mesh();

  std::ostringstream file_name_gmv;
  file_name_gmv << solution_type
                << ".out.gmv."
                << std::setw(2)
                << std::setfill('0')
                << std::right
                << a_step;

  GMVIO(mesh).write_equation_systems
    (file_name_gmv.str(), es);
#endif
}

// Set the parameters for the nonlinear and linear solvers to be used during the simulation
void set_system_parameters(StokesConvDiffSys &system, FEMParameters &param)
{
  // Use analytical jacobians?
  system.analytic_jacobians() = param.analytic_jacobians;

  // Verify analytic jacobians against numerical ones?
  system.verify_analytic_jacobians = param.verify_analytic_jacobians;

  // Use the prescribed FE type
  //system.fe_family() = param.fe_family[0];
  //system.fe_order() = param.fe_order[0];

  // More desperate debugging options
  system.print_solution_norms = param.print_solution_norms;
  system.print_solutions      = param.print_solutions;
  system.print_residual_norms = param.print_residual_norms;
  system.print_residuals      = param.print_residuals;
  system.print_jacobian_norms = param.print_jacobian_norms;
  system.print_jacobians      = param.print_jacobians;

  // No transient time solver
  system.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system));

  // Nonlinear solver options
  {
    NewtonSolver *solver = new NewtonSolver(system);
    system.time_solver->diff_solver() = AutoPtr<DiffSolver>(solver);

    solver->quiet                       = param.solver_quiet;
    solver->max_nonlinear_iterations    = param.max_nonlinear_iterations;
    solver->minsteplength               = param.min_step_length;
    solver->relative_step_tolerance     = param.relative_step_tolerance;
    solver->relative_residual_tolerance = param.relative_residual_tolerance;
    solver->require_residual_reduction  = param.require_residual_reduction;
    solver->linear_tolerance_multiplier = param.linear_tolerance_multiplier;
    if (system.time_solver->reduce_deltat_on_diffsolver_failure)
      {
        solver->continue_after_max_iterations = true;
        solver->continue_after_backtrack_failure = true;
      }

    // And the linear solver options
    solver->max_linear_iterations       = param.max_linear_iterations;
    solver->initial_linear_tolerance    = param.initial_linear_tolerance;
    solver->minimum_linear_tolerance    = param.minimum_linear_tolerance;
  }
}

// Build the mesh refinement object and set parameters for refining/coarsening etc

#ifdef LIBMESH_ENABLE_AMR

AutoPtr<MeshRefinement> build_mesh_refinement(MeshBase &mesh,
                                              FEMParameters &param)
{
  AutoPtr<MeshRefinement> mesh_refinement(new MeshRefinement(mesh));
  mesh_refinement->coarsen_by_parents() = true;
  mesh_refinement->absolute_global_tolerance() = param.global_tolerance;
  mesh_refinement->nelem_target()      = param.nelem_target;
  mesh_refinement->refine_fraction()   = param.refine_fraction;
  mesh_refinement->coarsen_fraction()  = param.coarsen_fraction;
  mesh_refinement->coarsen_threshold() = param.coarsen_threshold;

  return mesh_refinement;
}

#endif // LIBMESH_ENABLE_AMR

// This is where we declare the error estimators to be built and used for
// mesh refinement. The adjoint residual estimator needs two estimators.
// One for the forward component of the estimate and one for the adjoint
// weighting factor. Here we use the Patch Recovery indicator to estimate both the
// forward and adjoint weights. The H1 seminorm component of the error is used
// as dictated by the weak form the Laplace equation.

AutoPtr<ErrorEstimator> build_error_estimator(FEMParameters &param, QoISet &qois)
{
  AutoPtr<ErrorEstimator> error_estimator;

  if (param.indicator_type == "kelly")
    {
      std::cout<<"Using Kelly Error Estimator"<<std::endl;

      error_estimator.reset(new KellyErrorEstimator);
    }
  else if (param.indicator_type == "adjoint_residual")
    {
      std::cout<<"Using Adjoint Residual Error Estimator with Patch Recovery Weights"<<std::endl;

      AdjointResidualErrorEstimator *adjoint_residual_estimator = new AdjointResidualErrorEstimator;

      error_estimator.reset (adjoint_residual_estimator);

      adjoint_residual_estimator->qoi_set() = qois;

      adjoint_residual_estimator->error_plot_suffix = "error.gmv";

      PatchRecoveryErrorEstimator *p1 =
        new PatchRecoveryErrorEstimator;
      adjoint_residual_estimator->primal_error_estimator().reset(p1);

      PatchRecoveryErrorEstimator *p2 =
        new PatchRecoveryErrorEstimator;
      adjoint_residual_estimator->dual_error_estimator().reset(p2);

      adjoint_residual_estimator->primal_error_estimator()->error_norm.set_type(0, H1_SEMINORM);
      p1->set_patch_reuse(param.patch_reuse);

      adjoint_residual_estimator->dual_error_estimator()->error_norm.set_type(0, H1_SEMINORM);
      p2->set_patch_reuse(param.patch_reuse);
    }
  else
    libmesh_error_msg("Unknown indicator_type = " << param.indicator_type);

  return error_estimator;
}

// **********************************************************************
// The main program.
int main (int argc, char** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // Skip adaptive examples on a non-adaptive libMesh build
#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#else

  std::cout << "Started " << argv[0] << std::endl;

  // Make sure the general input file exists, and parse it
  {
    std::ifstream i("general.in");
    if (!i)
      libmesh_error_msg('[' << init.comm().rank() << "] Can't find general.in; exiting early.");
  }
  GetPot infile("general.in");

  // Read in parameters from the input file
  FEMParameters param;
  param.read(infile);

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());

  // And an object to refine it
  AutoPtr<MeshRefinement> mesh_refinement =
    build_mesh_refinement(mesh, param);

  // And an EquationSystems to run on it
  EquationSystems equation_systems (mesh);

  std::cout << "Building the mesh" << std::endl;

  MeshTools::Generation::build_square (mesh,
                                         20,
                                         20,
                                         0., 1.,
                                         0., 1.,
                                         QUAD9);

  // Create a mesh refinement object to do the initial uniform refinements
  //MeshRefinement initial_uniform_refinements(mesh);
  //initial_uniform_refinements.uniformly_refine(param.coarserefinements);

  std::cout << "Building system" << std::endl;

  // Build the FEMSystem
  StokesConvDiffSys &system = equation_systems.add_system<StokesConvDiffSys> ("StokesConvDiffSys");

  // Set its parameters
  set_system_parameters(system, param);

  std::cout << "Initializing systems" << std::endl;

  equation_systems.init ();

  // Print information about the mesh and system to the screen.
  mesh.print_info();
  equation_systems.print_info();
  LinearSolver<Number> *linear_solver = system.get_linear_solver();

  {
    // Adaptively solve the timestep
    unsigned int a_step = 0;
    for (; a_step != param.max_adaptivesteps; ++a_step)
      {
        // We can't adapt to both a tolerance and a
        // target mesh size
        if (param.global_tolerance != 0.)
          libmesh_assert_equal_to (param.nelem_target, 0);
        // If we aren't adapting to a tolerance we need a
        // target mesh size
        else
          libmesh_assert_greater (param.nelem_target, 0);

        linear_solver->reuse_preconditioner(false); //can reuse for adjoint, but not for new forwards solve

        // Solve the forward problem
        system.solve();

        // Write out the computed primal solution
        write_output(equation_systems, a_step, "primal");

        // Get a pointer to the primal solution vector
        NumericVector<Number> &primal_solution = *system.solution;

        // Declare a QoISet object, we need this object to set weights for our QoI error contributions
        QoISet qois;

        // Declare a qoi_indices vector, each index will correspond to a QoI
        std::vector<unsigned int> qoi_indices;
        qoi_indices.push_back(0);
        qois.add_indices(qoi_indices);

        // Set weights for each index, these will weight the contribution of each QoI in the final error
        // estimate to be used for flagging elements for refinement
        qois.set_weight(0, 1.0);

				// A SensitivityData object to hold the qois and parameters
        SensitivityData sensitivities(qois, system, system.get_parameter_vector());

        // Make sure we get the contributions to the adjoint RHS from the sides
        system.assemble_qoi_sides = true; //does nothing anyways...

        // We are about to solve the adjoint system, but before we do this we see the same preconditioner
        // flag to reuse the preconditioner from the forward solver
        linear_solver->reuse_preconditioner(param.reuse_preconditioner);

        // Here we solve the adjoint problem inside the adjoint_qoi_parameter_sensitivity
        // function, so we have to set the adjoint_already_solved boolean to false
        system.set_adjoint_already_solved(false);

        // Compute the sensitivities
        system.adjoint_qoi_parameter_sensitivity(qois, system.get_parameter_vector(), sensitivities);

        // Now that we have solved the adjoint, set the adjoint_already_solved boolean to true, 
        //so we dont solve unneccesarily in the error estimator
        system.set_adjoint_already_solved(true);
        
        Number sensit = sensitivities[0][0];
        std::cout << "Sensitivity of QoI to Peclet number is " << sensit << std::endl;

        // Get a pointer to the solution vector of the adjoint problem for QoI 0
        NumericVector<Number> &dual_solution = system.get_adjoint_solution(0);

        // Swap the (pointers to) primal and dual solutions so we can write out the adjoint solution
        primal_solution.swap(dual_solution);
        write_output(equation_systems, a_step, "adjoint");

        // Swap back
        primal_solution.swap(dual_solution);

        std::cout << "Adaptive step " << a_step << ", we have " << mesh.n_active_elem()
                  << " active elements and "
                  << equation_systems.n_active_dofs()
                  << " active dofs." << std::endl ;

        // Postprocess, compute the approximate QoIs and write them out to the console
        std::cout << "Postprocessing: " << std::endl;
        system.postprocess_sides = false; //QoI doesn't involve edges
        system.postprocess();
        Number QoI_computed = system.get_QoI_value("computed", 0);

        std::cout<< "Computed QoI is " << std::setprecision(17)
                 << QoI_computed << std::endl;

        // Now we construct the data structures for the mesh refinement process
        ErrorVector error;

        // Build an error estimator object
        AutoPtr<ErrorEstimator> error_estimator =
          build_error_estimator(param, qois);

        // Estimate the error in each element using the Adjoint Residual or Kelly error estimator
        error_estimator->estimate_error(system, error);

        // We have to refine either based on reaching an error tolerance or
        // a number of elements target, which should be verified above
        // Otherwise we flag elements by error tolerance or nelem target

        // Uniform refinement
        if(param.refine_uniformly)
          {
            mesh_refinement->uniformly_refine(1);
          }
        // Adaptively refine based on reaching an error tolerance
        else if(param.global_tolerance >= 0. && param.nelem_target == 0.)
          {
            mesh_refinement->flag_elements_by_error_tolerance (error);

            mesh_refinement->refine_and_coarsen_elements();
          }
        // Adaptively refine based on reaching a target number of elements
        else
          {
            if (mesh.n_active_elem() >= param.nelem_target)
              {
                std::cout<<"We reached the target number of elements."<<std::endl <<std::endl;
                break;
              }

            mesh_refinement->flag_elements_by_nelem_target (error);

            mesh_refinement->refine_and_coarsen_elements();
          }

        // Dont forget to reinit the system after each adaptive refinement !
        equation_systems.reinit();

        std::cout << "Refined mesh to "
                  << mesh.n_active_elem()
                  << " active elements and "
                  << equation_systems.n_active_dofs()
                  << " active dofs." << std::endl;
      }

    // Do one last solve if necessary
    if (a_step == param.max_adaptivesteps)
      {
        linear_solver->reuse_preconditioner(false);
        system.solve();

        write_output(equation_systems, a_step, "primal");

        NumericVector<Number> &primal_solution = *system.solution;

        QoISet qois;
        std::vector<unsigned int> qoi_indices;

        qoi_indices.push_back(0);
        qois.add_indices(qoi_indices);
        qois.set_weight(0, 1.0);
        
        SensitivityData sensitivities(qois, system, system.get_parameter_vector());

        system.assemble_qoi_sides = false; //QoI doesn't involve sides
        linear_solver->reuse_preconditioner(param.reuse_preconditioner);
        
        system.set_adjoint_already_solved(false);

        system.adjoint_qoi_parameter_sensitivity(qois, system.get_parameter_vector(), sensitivities);

        // Now that we have solved the adjoint, set the adjoint_already_solved boolean to true, 
        //so we dont solve unneccesarily in the error estimator
        system.set_adjoint_already_solved(true);
        
        Number sensit = sensitivities[0][0];
        std::cout << "Sensitivity of QoI to Peclet number is " << sensit << std::endl;

        NumericVector<Number> &dual_solution = system.get_adjoint_solution(0);

        primal_solution.swap(dual_solution);
        write_output(equation_systems, a_step, "adjoint");
        primal_solution.swap(dual_solution);

        std::cout << "Adaptive step " << a_step << ", we have " << mesh.n_active_elem()
                  << " active elements and "
                  << equation_systems.n_active_dofs()
                  << " active dofs." << std::endl ;

        std::cout << "Postprocessing: " << std::endl;
        system.postprocess_sides = false; //nothing special on sides
        system.postprocess();

        Number QoI_computed = system.get_QoI_value("computed", 0);

        std::cout<< "Computed QoI is " << std::setprecision(17)
                 << QoI_computed << std::endl;
      }
  }

  std::cerr << '[' << mesh.processor_id()
            << "] Completing output." << std::endl;

#endif // #ifndef LIBMESH_ENABLE_AMR

  // All done.
  return 0;
}
