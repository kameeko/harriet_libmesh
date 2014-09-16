#include "libmesh/getpot.h"

#include "convdiffstokes_sys.h"

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

// System initialization
void StokesConvDiffSys::init_data (){
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
void StokesConvDiffSys::init_context(DiffContext &context){
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
bool StokesConvDiffSys::element_time_derivative (bool request_jacobian, DiffContext& context){

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
void StokesConvDiffSys::postprocess (){
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
Point StokesConvDiffSys::forcing(const Point& pt){
	Point f;
	//f(0) = exp(-10*(pow(pt(0)-0.25,2)+pow(pt(1)-0.25,2)));
	f(0) = 0.75**exp(-10*(pow(pt(0)-0.25,2)+pow(pt(1)-0.25,2)));
	return f;
}

