//reproducing steady coupled stokes-conveciton-diffusion with FEMSystem framework

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
//#include "stokes_convdiff_system.C" //linkage unhappy if things spread out...
#include "libmesh/fem_system.h"
#include "libmesh/boundary_info.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/mesh.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/zero_function.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;
class StokesConvDiffSys : public FEMSystem{

public:

  // Indices for each variable;
  unsigned int p_var, u_var, v_var, c_var;
  
  // Constructor
  StokesConvDiffSys(EquationSystems& es,
               const std::string& name_in,
               const unsigned int number_in)
    : FEMSystem(es, name_in, number_in){}

  // System initialization
  void init_data (){
		const unsigned int dim = this->get_mesh().mesh_dimension();

		//polynomial order and finite element type for pressure variable
		unsigned int pressure_p = 1;
		GetPot infile("stokes_convdiff.in");
		std::string fe_family = infile("fe_family", std::string("LAGRANGE"));

		// LBB needs better-than-quadratic velocities for better-than-linear
		// pressures, and libMesh needs non-Lagrange elements for
		// better-than-quadratic velocities.
		libmesh_assert((pressure_p == 1) || (fe_family != "LAGRANGE"));

		FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);

		// Add the velocity components "u" & "v".  They
		// will be approximated using second-order approximation.
		u_var = this->add_variable ("u", static_cast<Order>(pressure_p+1),
		                            fefamily);
		v_var = this->add_variable ("v",
		                            static_cast<Order>(pressure_p+1),
		                            fefamily);                  

		// Add the pressure variable "p". This will
		// be approximated with a first-order basis,
		// providing an LBB-stable pressure-velocity pair.
		p_var = this->add_variable ("p",
		                            static_cast<Order>(pressure_p),
		                            fefamily);
		                            
		//first-order for concentration, like in original implementation                            
		c_var = this->add_variable("c", static_cast<Order>(pressure_p), fefamily);          

		//indicate variables that change in time
		this->time_evolving(u_var);
		this->time_evolving(v_var);
		this->time_evolving(p_var);
		this->time_evolving(c_var);

		// Useful debugging options
		// Set verify_analytic_jacobians to 1e-6 to use
		this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
		this->print_jacobians = infile("print_jacobians", false);
		this->print_element_jacobians = infile("print_element_jacobians", false);

		// Set Dirichlet boundary conditions
		const boundary_id_type top_id = 2;

		std::set<boundary_id_type> top_bdys;
		top_bdys.insert(top_id);

		const boundary_id_type all_ids[6] = {0, 1, 2, 3, 4, 5};
		std::set<boundary_id_type> all_bdys(all_ids, all_ids+(dim*2));

		std::set<boundary_id_type> nontop_bdys = all_bdys;
		nontop_bdys.erase(top_id);

		std::vector<unsigned int> u_only(1, u_var);
		std::vector<unsigned int> v_only(1, v_var);
		std::vector<unsigned int> uv(1, u_var); uv.push_back(v_var);

		ZeroFunction<Number> zero;
		ConstFunction<Number> one(1);
		
		// For lid-driven cavity, set u=1,v=0 on the lid and u=v=0 elsewhere
		this->get_dof_map().add_dirichlet_boundary
		  (DirichletBoundary (top_bdys, u_only, &one));
		this->get_dof_map().add_dirichlet_boundary
		  (DirichletBoundary (top_bdys, v_only, &zero));
		this->get_dof_map().add_dirichlet_boundary
		  (DirichletBoundary (nontop_bdys, uv, &zero));
		  
		// Do the parent's initialization after variables and boundary constraints are defined
		FEMSystem::init_data();
  }

  // Context initialization
  void init_context(DiffContext &context){
		FEMContext &ctxt = cast_ref<FEMContext&>(context);

		FEBase* u_elem_fe;
		FEBase* p_elem_fe;
		FEBase* c_elem_fe;
		FEBase* u_side_fe; //why does only u have one of these?

		ctxt.get_element_fe( u_var, u_elem_fe );
		ctxt.get_element_fe( p_var, p_elem_fe );
		ctxt.get_element_fe(c_var, c_elem_fe);
		ctxt.get_side_fe( u_var, u_side_fe );

		// We should prerequest all the data
		// we will need to build the linear system.
		u_elem_fe->get_JxW();
		u_elem_fe->get_phi();
		u_elem_fe->get_dphi();
		u_elem_fe->get_xyz();

		p_elem_fe->get_phi();
		p_elem_fe->get_xyz();

		u_side_fe->get_JxW();
		u_side_fe->get_phi();
		u_side_fe->get_xyz();
		
		c_elem_fe->get_dphi();
		c_elem_fe->get_xyz();
  }

  // Element residual and jacobian calculations
  // Time dependent parts
  bool element_time_derivative (bool request_jacobian, DiffContext& context){

		FEMContext &ctxt = cast_ref<FEMContext&>(context);

		FEBase* u_elem_fe;
		FEBase* p_elem_fe;
		FEBase* c_elem_fe;

		ctxt.get_element_fe( u_var, u_elem_fe );
		ctxt.get_element_fe( p_var, p_elem_fe );
		ctxt.get_element_fe(c_var, c_elem_fe);

		// First we get some references to cell-specific data that
		// will be used to assemble the linear system.

		// Element Jacobian * quadrature weights for interior integration
		const std::vector<Real> &JxW = u_elem_fe->get_JxW();

		//for velocities, at interior quadrature points
		const std::vector<std::vector<Real> >& phi = u_elem_fe->get_phi();
		const std::vector<std::vector<RealGradient> >& dphi = u_elem_fe->get_dphi();

		//for pressure and concentration, at interior quadrature points
		const std::vector<std::vector<Real> >& psi = p_elem_fe->get_phi();
		const std::vector<std::vector<RealGradient> >& dpsi = c_elem_fe->get_dphi();

		// Physical location of the quadrature points
		const std::vector<Point>& qpoint = u_elem_fe->get_xyz();

		// The number of local degrees of freedom in each variable
		const unsigned int n_p_dofs = ctxt.get_dof_indices( p_var ).size();
		const unsigned int n_u_dofs = ctxt.get_dof_indices( u_var ).size();
		const unsigned int n_c_dofs = ctxt.get_dof_indices(c_var).size();
		libmesh_assert_equal_to (n_u_dofs, ctxt.get_dof_indices( v_var ).size());

		// The subvectors and submatrices we need to fill:
		const unsigned int dim = this->get_mesh().mesh_dimension();
		DenseSubMatrix<Number> &Kuu = ctxt.get_elem_jacobian( u_var, u_var );
		DenseSubMatrix<Number> &Kvv = ctxt.get_elem_jacobian( v_var, v_var );
		DenseSubMatrix<Number> &Kpp = ctxt.get_elem_jacobian(p_var, p_var);
		DenseSubMatrix<Number> &Kcc = ctxt.get_elem_jacobian(c_var, c_var);
		DenseSubMatrix<Number> &Kuv = ctxt.get_elem_jacobian( u_var, v_var );
		DenseSubMatrix<Number> &Kup = ctxt.get_elem_jacobian( u_var, p_var );
		DenseSubMatrix<Number> &Kuc = ctxt.get_elem_jacobian( u_var, c_var );
		DenseSubMatrix<Number> &Kvu = ctxt.get_elem_jacobian( v_var, u_var ); 
		DenseSubMatrix<Number> &Kvp = ctxt.get_elem_jacobian( v_var, p_var );
		DenseSubMatrix<Number> &Kvc = ctxt.get_elem_jacobian( v_var, c_var );
		DenseSubMatrix<Number> &Kpu = ctxt.get_elem_jacobian( p_var, u_var );
		DenseSubMatrix<Number> &Kpv = ctxt.get_elem_jacobian( p_var, v_var );
		DenseSubMatrix<Number> &Kpc = ctxt.get_elem_jacobian( p_var, c_var );
		DenseSubMatrix<Number> &Kcu = ctxt.get_elem_jacobian( c_var, u_var );
		DenseSubMatrix<Number> &Kcv = ctxt.get_elem_jacobian( c_var, v_var );
		DenseSubMatrix<Number> &Kcp = ctxt.get_elem_jacobian( c_var, p_var );
		DenseSubVector<Number> &Fu = ctxt.get_elem_residual( u_var );
		DenseSubVector<Number> &Fv = ctxt.get_elem_residual( v_var );
		DenseSubVector<Number> &Fp = ctxt.get_elem_residual( p_var );
		DenseSubVector<Number> &Fc = ctxt.get_elem_residual( c_var );

		// Now we will build the element Jacobian and residual.
		// Constructing the residual requires the solution and its
		// gradient from the previous timestep.  This must be
		// calculated at each quadrature point by summing the
		// solution degree-of-freedom values by the appropriate
		// weight functions.
		unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

		for (unsigned int qp=0; qp != n_qpoints; qp++)
		  {
		    // Compute the solution & its gradient at the old Newton iterate
		    Number p = ctxt.interior_value(p_var, qp),
		      u = ctxt.interior_value(u_var, qp),
		      v = ctxt.interior_value(v_var, qp),
		      c = ctxt.interior_value(c_var, qp);
		    Gradient grad_u = ctxt.interior_gradient(u_var, qp),
		      grad_v = ctxt.interior_gradient(v_var, qp),
		      grad_c = ctxt.interior_gradient(c_var, qp);

		    // Definitions for convenience.  It is sometimes simpler to do a
		    // dot product if you have the full vector at your disposal.
		    NumberVectorValue U     (u,     v);

		    const Number  u_x = grad_u(0);
		    const Number  v_y = grad_v(1);
		    const Number c_x = grad_c(0);
		    const Number c_y = grad_c(1);

		    // Value of the forcing function at this quadrature point
		    Point f = this->forcing(qpoint[qp]);

		    // First, an i-loop over the velocity degrees of freedom.
		    // We know that n_u_dofs == n_v_dofs so we can compute contributions
		    // for both at the same time.
		    for (unsigned int i=0; i != n_u_dofs; i++){ 
		      Fu(i) += JxW[qp] * (p*dphi[i][qp](0) - (grad_u*dphi[i][qp]));
		      Fv(i) += JxW[qp] * (p*dphi[i][qp](1) - (grad_v*dphi[i][qp]));

		      if (request_jacobian && ctxt.elem_solution_derivative){
		        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

		        // Matrix contributions for the uu and vv couplings.
		        for (unsigned int j=0; j != n_u_dofs; j++){ 
		        	Kuu(i,j) += JxW[qp]*(-dphi[j][qp]*dphi[i][qp]);
		        	Kvv(i,j) += JxW[qp]*(-dphi[j][qp]*dphi[i][qp]);
		        }

		        // Matrix contributions for the up and vp couplings.
		        for (unsigned int j=0; j != n_p_dofs; j++){
		          Kup(i,j) += JxW[qp]*psi[j][qp]*dphi[i][qp](0);
		          Kvp(i,j) += JxW[qp]*psi[j][qp]*dphi[i][qp](1);
		        }
		      }
		    }
		    
		    //loop over p and c degrees of freedom
		    for (unsigned int i=0; i != n_p_dofs; i++){ 
		      Fp(i) += JxW[qp] * (u_x*psi[i][qp] + v_y*psi[i][qp]);
		      Fc(i) += JxW[qp] * (-grad_c*dpsi[i][qp] + U*grad_c*psi[i][qp] + f(0)*psi[i][qp]);
		      
		      if (request_jacobian && ctxt.elem_solution_derivative){
		        for(unsigned int j=0; j != n_u_dofs; j++){
		          Kpu(i,j) += JxW[qp]*(dphi[j][qp](0)*psi[i][qp]);
		        	Kpv(i,j) += JxW[qp]*(dphi[j][qp](1)*psi[i][qp]);
		        	Kcu(i,j) += JxW[qp]*(phi[j][qp]*c_x*psi[i][qp]);
		        	Kcv(i,j) += JxW[qp]*(phi[j][qp]*c_y*psi[i][qp]);
		       	}
		       	for(unsigned int j=0; j != n_c_dofs; j++){
		       		Kcc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp]);
		       	}
		      }
		    }
		  } // end of the quadrature point qp-loop

		return request_jacobian;
  }

  // Postprocessed output
  void postprocess (){
		const unsigned int dim = this->get_mesh().mesh_dimension();

		Point pt(1./3., 1./3.);
		Number u = point_value(u_var, pt),
		  v = point_value(v_var, pt),
		  p = point_value(p_var, pt),
		  c = point_value(c_var, pt);

		std::cout << "u(1/3,1/3) = ("
		          << u << ", "
		          << v << ", "
		          << ")" << std::endl;
		std::cout << "p(1/3,1/3) = (" << p << ")" << std::endl;
		std::cout << "c(1/3,1/3) = (" << c << ")" << std::endl;
	}

  // Returns the value of a forcing function at point p.  This value
  // depends on which application is being used.
  Point forcing(const Point& pt){
		Point f;
		//f(0) = exp(-10*(pow(pt(0)-0.25,2)+pow(pt(1)-0.25,2)));
		f(0)=0.75*exp(-10*(pow(pt(0)-0.25,2)+pow(pt(1)-0.25,2)));
		return f;
  }
  
}; //end StokesConvDiffSys class definition



int main(int argc, char** argv){

	//initialize libMesh
	LibMeshInit init(argc, argv);
	
	//parameters
	GetPot infile("fem_system_params.in");
  const Real global_tolerance          = infile("global_tolerance", 0.);
  const unsigned int nelem_target      = infile("n_elements", 400);
  const bool transient                 = infile("transient", true);
  const Real deltat                    = infile("deltat", 0.005);
  unsigned int n_timesteps             = infile("n_timesteps", 20);
  const unsigned int coarsegridsize    = infile("coarsegridsize", 1);
  const unsigned int coarserefinements = infile("coarserefinements", 0);
  const unsigned int max_adaptivesteps = infile("max_adaptivesteps", 10);
  const unsigned int dim               = 2;
  
#ifdef LIBMESH_HAVE_EXODUS_API
  const unsigned int write_interval    = infile("write_interval", 5);
#endif

  // Create a mesh, with dimension to be overridden later, distributed
  // across the default MPI communicator.
  Mesh mesh(init.comm());

  // And an object to refine it
  MeshRefinement mesh_refinement(mesh);
  mesh_refinement.coarsen_by_parents() = true;
  mesh_refinement.absolute_global_tolerance() = global_tolerance;
  mesh_refinement.nelem_target() = nelem_target;
  mesh_refinement.refine_fraction() = 0.3;
  mesh_refinement.coarsen_fraction() = 0.3;
  mesh_refinement.coarsen_threshold() = 0.1;

	MeshTools::Generation::build_square (mesh,
                                         coarsegridsize,
                                         coarsegridsize,
                                         0., 1.,
                                         0., 1.,
                                         QUAD9);
  mesh_refinement.uniformly_refine(coarserefinements);
  
  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);
  
  //name system
  StokesConvDiffSys & system = 
  	equation_systems.add_system<StokesConvDiffSys>("StokesConvDiff");
  	
 	system.time_solver =
    AutoPtr<TimeSolver>(new SteadySolver(system));
  libmesh_assert_equal_to (n_timesteps, 1);
  
  // Initialize the system
  equation_systems.init ();

  // Set the time stepping options
  system.deltat = deltat;

  // And the nonlinear solver options
  //DiffSolver &solver = *(system.time_solver->diff_solver().get());
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

  // And the linear solver options
  solver->max_linear_iterations =
    infile("max_linear_iterations", 50000);
  solver->initial_linear_tolerance =
    infile("initial_linear_tolerance", 1.e-3);

  // Print information about the system to the screen.
  equation_systems.print_info();

  // Now we begin the timestep loop to compute the time-accurate
  // solution of the equations...not that this is transient, but eh, why not...
	for (unsigned int t_step=0; t_step != n_timesteps; ++t_step){
    // A pretty update message
    std::cout << "\n\nSolving time step " << t_step << ", time = "
              << system.time << std::endl;

    // Adaptively solve the timestep
    unsigned int a_step = 0;
    for (; a_step != max_adaptivesteps; ++a_step)
      {

        system.solve();

        system.postprocess();

        ErrorVector error;

        AutoPtr<ErrorEstimator> error_estimator;

        // To solve to a tolerance in this problem we
        // need a better estimator than Kelly
        if (global_tolerance != 0.)
          {
            // We can't adapt to both a tolerance and a mesh
            // size at once
            libmesh_assert_equal_to (nelem_target, 0);

            UniformRefinementEstimator *u =
              new UniformRefinementEstimator;

            // The lid-driven cavity problem isn't in H1, so
            // lets estimate L2 error
            u->error_norm = L2;

            error_estimator.reset(u);
          }
        else
          {
            // If we aren't adapting to a tolerance we need a
            // target mesh size
            libmesh_assert_greater (nelem_target, 0);

            // Kelly is a lousy estimator to use for a problem
            // not in H1 - if we were doing more than a few
            // timesteps we'd need to turn off or limit the
            // maximum level of our adaptivity eventually
            error_estimator.reset(new KellyErrorEstimator);
          }

        // Calculate error
        std::vector<Real> weights(4,1.0);  // based on u, v, p, c

        // Keep the same default norm type.
        std::vector<FEMNormType>
          norms(1, error_estimator->error_norm.type(0));
        error_estimator->error_norm = SystemNorm(norms, weights);

        error_estimator->estimate_error(system, error);

        // Print out status at each adaptive step.
        Real global_error = error.l2_norm();
        std::cout << "Adaptive step " << a_step << ": " << std::endl;
        if (global_tolerance != 0.)
          std::cout << "Global_error = " << global_error
                    << std::endl;
        if (global_tolerance != 0.)
          std::cout << "Worst element error = " << error.maximum()
                    << ", mean = " << error.mean() << std::endl;

        if (global_tolerance != 0.)
          {
            // If we've reached our desired tolerance, we
            // don't need any more adaptive steps
            if (global_error < global_tolerance)
              break;
            mesh_refinement.flag_elements_by_error_tolerance(error);
          }
        else
          {
            // If flag_elements_by_nelem_target returns true, this
            // should be our last adaptive step.
            if (mesh_refinement.flag_elements_by_nelem_target(error))
              {
                mesh_refinement.refine_and_coarsen_elements();
                equation_systems.reinit();
                a_step = max_adaptivesteps;
                break;
              }
          }

        // Carry out the adaptive mesh refinement/coarsening
        mesh_refinement.refine_and_coarsen_elements();
        equation_systems.reinit();

        std::cout << "Refined mesh to "
                  << mesh.n_active_elem()
                  << " active elements and "
                  << equation_systems.n_active_dofs()
                  << " active dofs." << std::endl;
      }
    // Do one last solve if necessary
    if (a_step == max_adaptivesteps)
      {
        system.solve();

        system.postprocess();
      }

    // Advance to the next timestep in a transient problem
    system.time_solver->advance_timestep();

#ifdef LIBMESH_HAVE_EXODUS_API
    // Write out this timestep if we're requested to
    if ((t_step+1)%write_interval == 0)
      {
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
