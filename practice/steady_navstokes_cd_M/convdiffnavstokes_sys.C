#include "libmesh/getpot.h"

#include "convdiffnavstokes_sys.h"

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
void NavStokesConvDiffSys::init_data (){
	const unsigned int dim = this->get_mesh().mesh_dimension();

	//polynomial order and finite element type for pressure variable
	unsigned int pressure_p = 1;
	GetPot infile("navstokes_convdiff.in");
	std::string fe_family = infile("fe_family", std::string("LAGRANGE"));
	
	rho = infile("rho", 1.0);
	mu = infile("mu", 1.0);
	params.push_back(rho);
	params.push_back(mu);

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
	
	//adjoint variables in same space as their...primal? counterparts...
	zu_var = this->add_variable("zu", static_cast<Order>(pressure_p+1), fefamily);    
	zv_var = this->add_variable("zv", static_cast<Order>(pressure_p+1), fefamily);  
	zp_var = this->add_variable("zp", static_cast<Order>(pressure_p), fefamily);  
	zc_var = this->add_variable("zc", static_cast<Order>(pressure_p), fefamily); 
	
	//source parameter in same space as the thing it spews...
	fc_var = this->add_variable("fc", static_cast<Order>(pressure_p), fefamily);  
	
	//auxillary variables
	aux_u_var = this->add_variable("aux_u", static_cast<Order>(pressure_p+1), fefamily);   
	aux_v_var = this->add_variable("aux_v", static_cast<Order>(pressure_p+1), fefamily); 
	aux_p_var = this->add_variable("aux_p", static_cast<Order>(pressure_p), fefamily); 
	aux_c_var = this->add_variable("aux_c", static_cast<Order>(pressure_p), fefamily); 
	aux_zu_var = this->add_variable("aux_zu", static_cast<Order>(pressure_p+1), fefamily); 
	aux_zv_var = this->add_variable("aux_zv", static_cast<Order>(pressure_p+1), fefamily); 
	aux_zp_var = this->add_variable("aux_zp", static_cast<Order>(pressure_p), fefamily); 
	aux_zc_var = this->add_variable("aux_zc", static_cast<Order>(pressure_p), fefamily); 
	aux_fc_var = this->add_variable("aux_fc", static_cast<Order>(pressure_p), fefamily);  

	//regularization
	beta = infile("beta",0.1);
	regtype = infile("regularization_option",1);

	//indicate variables that change in time
	this->time_evolving(u_var);
	this->time_evolving(v_var);
	this->time_evolving(p_var);
	this->time_evolving(c_var);
	this->time_evolving(zu_var);
	this->time_evolving(zv_var);
	this->time_evolving(zp_var);
	this->time_evolving(zc_var);
	this->time_evolving(fc_var);
	this->time_evolving(aux_u_var);
	this->time_evolving(aux_v_var);
	this->time_evolving(aux_p_var);
	this->time_evolving(aux_c_var);
	this->time_evolving(aux_zu_var);
	this->time_evolving(aux_zv_var);
	this->time_evolving(aux_zp_var);
	this->time_evolving(aux_zc_var);
	this->time_evolving(aux_fc_var);

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
	std::vector<unsigned int> c_only(1, c_var);
	std::vector<unsigned int> z_forDiri(1, zu_var);
		z_forDiri.push_back(zv_var); z_forDiri.push_back(zc_var);
	std::vector<unsigned int> aux_forDiri(1, aux_u_var);
		aux_forDiri.push_back(aux_v_var); aux_forDiri.push_back(aux_c_var); 
		aux_forDiri.push_back(aux_zu_var); aux_forDiri.push_back(aux_zv_var); 
		aux_forDiri.push_back(aux_zc_var); 

	ZeroFunction<Number> zero;
	ConstFunction<Number> one(1);
	
	// For lid-driven cavity, set u=1,v=0 on the lid and u=v=0 elsewhere
	this->get_dof_map().add_dirichlet_boundary
	  (DirichletBoundary (top_bdys, u_only, &one));
	this->get_dof_map().add_dirichlet_boundary
	  (DirichletBoundary (top_bdys, v_only, &zero));
	this->get_dof_map().add_dirichlet_boundary
	  (DirichletBoundary (nontop_bdys, uv, &zero));
	  
	//c=0 on boundary, cuz I feel like it...
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, c_only, &zero));
	
	//corresponding BCs for adjoints
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, z_forDiri, &zero));
	
	//corresponding BCs for auxillary variables
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, aux_forDiri, &zero));
	
	if(regtype == 1){
		std::vector<unsigned int> fc_only(1, fc_var); fc_only.push_back(aux_fc_var);
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, fc_only, &zero));
	}
	  
	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

// Context initialization
void NavStokesConvDiffSys::init_context(DiffContext &context){
	FEMContext &ctxt = cast_ref<FEMContext&>(context);

	//stuff for things of velocity's family
	FEBase* u_elem_fe = NULL; //initializing pointer

  ctxt.get_element_fe(u_var, u_elem_fe);
  u_elem_fe->get_JxW();
  u_elem_fe->get_phi();
  u_elem_fe->get_dphi();

  FEBase* u_side_fe = NULL;
  ctxt.get_side_fe(u_var, u_side_fe);

  u_side_fe->get_JxW();
  u_side_fe->get_phi();
  u_side_fe->get_dphi();
  
	//stuff for things of pressure's family
	FEBase* p_elem_fe = NULL;

  ctxt.get_element_fe(p_var, p_elem_fe);
  p_elem_fe->get_JxW();
  p_elem_fe->get_phi();
  p_elem_fe->get_dphi();

  FEBase* p_side_fe = NULL;
  ctxt.get_side_fe(p_var, p_side_fe);

  p_side_fe->get_JxW();
  p_side_fe->get_phi();
  p_side_fe->get_dphi();
}

// Element residual and jacobian calculations
// Time dependent parts
bool NavStokesConvDiffSys::element_time_derivative (bool request_jacobian, DiffContext& context){

	FEMContext &ctxt = cast_ref<FEMContext&>(context);

	//some cell-specific stuff, for u's family and p's family
	FEBase* u_elem_fe = NULL; 
  ctxt.get_element_fe( u_var, u_elem_fe );
  FEBase* p_elem_fe = NULL; 
  ctxt.get_element_fe( p_var, p_elem_fe );

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = u_elem_fe->get_JxW();

	//for velocities, at interior quadrature points
	const std::vector<std::vector<Real> >& phi = u_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi = u_elem_fe->get_dphi();

	//for pressure and concentration, at interior quadrature points
	const std::vector<std::vector<Real> >& psi = p_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dpsi = p_elem_fe->get_dphi();

	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = u_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_p_dofs = ctxt.get_dof_indices( p_var ).size();
	const unsigned int n_u_dofs = ctxt.get_dof_indices( u_var ).size();
	libmesh_assert_equal_to (n_u_dofs, ctxt.get_dof_indices( v_var ).size());

	// The subvectors and submatrices we need to fill:
	const unsigned int dim = this->get_mesh().mesh_dimension();
	DenseSubMatrix<Number> &J_u_u = ctxt.get_elem_jacobian( u_var, u_var );
	DenseSubMatrix<Number> &J_u_v = ctxt.get_elem_jacobian( u_var, v_var );
	DenseSubMatrix<Number> &J_u_zu = ctxt.get_elem_jacobian( u_var, zu_var );
	DenseSubMatrix<Number> &J_u_zv = ctxt.get_elem_jacobian( u_var, zv_var );
	DenseSubMatrix<Number> &J_u_auxu = ctxt.get_elem_jacobian( u_var, aux_u_var );
	DenseSubMatrix<Number> &J_u_auxv = ctxt.get_elem_jacobian( u_var, aux_v_var );
	DenseSubMatrix<Number> &J_u_auxzu = ctxt.get_elem_jacobian( u_var, aux_zu_var );
	DenseSubMatrix<Number> &J_u_auxzv = ctxt.get_elem_jacobian( u_var, aux_zv_var );
	DenseSubMatrix<Number> &J_u_c = ctxt.get_elem_jacobian( u_var, c_var );
	DenseSubMatrix<Number> &J_u_zc = ctxt.get_elem_jacobian( u_var, zc_var );
	DenseSubMatrix<Number> &J_u_auxzc = ctxt.get_elem_jacobian( u_var, aux_zc_var );
	DenseSubMatrix<Number> &J_u_auxzp = ctxt.get_elem_jacobian( u_var, aux_zp_var );
	DenseSubMatrix<Number> &J_u_auxc = ctxt.get_elem_jacobian( u_var, aux_c_var );

	DenseSubMatrix<Number> &J_v_u = ctxt.get_elem_jacobian( v_var, u_var );
	DenseSubMatrix<Number> &J_v_v = ctxt.get_elem_jacobian( v_var, v_var );
	DenseSubMatrix<Number> &J_v_zu = ctxt.get_elem_jacobian( v_var, zu_var );
	DenseSubMatrix<Number> &J_v_zv = ctxt.get_elem_jacobian( v_var, zv_var );
	DenseSubMatrix<Number> &J_v_auxu = ctxt.get_elem_jacobian( v_var, aux_u_var );
	DenseSubMatrix<Number> &J_v_auxv = ctxt.get_elem_jacobian( v_var, aux_v_var );
	DenseSubMatrix<Number> &J_v_auxzu = ctxt.get_elem_jacobian( v_var, aux_zu_var );
	DenseSubMatrix<Number> &J_v_auxzv = ctxt.get_elem_jacobian( v_var, aux_zv_var );
	DenseSubMatrix<Number> &J_v_c = ctxt.get_elem_jacobian( v_var, c_var );
	DenseSubMatrix<Number> &J_v_zc = ctxt.get_elem_jacobian( v_var, zc_var );
	DenseSubMatrix<Number> &J_v_auxzc = ctxt.get_elem_jacobian( v_var, aux_zc_var );
	DenseSubMatrix<Number> &J_v_auxzp = ctxt.get_elem_jacobian( v_var, aux_zp_var );
	DenseSubMatrix<Number> &J_v_auxc = ctxt.get_elem_jacobian( v_var, aux_c_var );
	
	DenseSubMatrix<Number> &J_p_auxzu = ctxt.get_elem_jacobian( p_var, aux_zu_var );
	DenseSubMatrix<Number> &J_p_auxzv = ctxt.get_elem_jacobian( p_var, aux_zv_var );
	
	DenseSubMatrix<Number> &J_c_u = ctxt.get_elem_jacobian(c_var, u_var);
	DenseSubMatrix<Number> &J_c_v = ctxt.get_elem_jacobian(c_var, v_var);
	DenseSubMatrix<Number> &J_c_c = ctxt.get_elem_jacobian(c_var, c_var); 
	DenseSubMatrix<Number> &J_c_zc = ctxt.get_elem_jacobian(c_var, zc_var);
	DenseSubMatrix<Number> &J_c_auxzc = ctxt.get_elem_jacobian(c_var, aux_zc_var);
	DenseSubMatrix<Number> &J_c_auxu = ctxt.get_elem_jacobian(c_var, aux_u_var);
	DenseSubMatrix<Number> &J_c_auxv = ctxt.get_elem_jacobian(c_var, aux_v_var);
	DenseSubMatrix<Number> &J_c_auxc = ctxt.get_elem_jacobian(c_var, aux_c_var);
	
	DenseSubMatrix<Number> &J_zu_u = ctxt.get_elem_jacobian(zu_var, u_var);
	DenseSubMatrix<Number> &J_zu_v = ctxt.get_elem_jacobian(zu_var, v_var);
	DenseSubMatrix<Number> &J_zu_auxu = ctxt.get_elem_jacobian(zu_var, aux_u_var);
	DenseSubMatrix<Number> &J_zu_auxp = ctxt.get_elem_jacobian(zu_var, aux_p_var);
	
	DenseSubMatrix<Number> &J_zv_u = ctxt.get_elem_jacobian(zv_var, u_var);
	DenseSubMatrix<Number> &J_zv_v = ctxt.get_elem_jacobian(zv_var, v_var);
	DenseSubMatrix<Number> &J_zv_auxv = ctxt.get_elem_jacobian(zv_var, aux_v_var);
	DenseSubMatrix<Number> &J_zv_auxp = ctxt.get_elem_jacobian(zv_var, aux_p_var);
	
	DenseSubMatrix<Number> &J_zp_auxu = ctxt.get_elem_jacobian(zp_var, aux_u_var);
	DenseSubMatrix<Number> &J_zp_auxv = ctxt.get_elem_jacobian(zp_var, aux_v_var);
	
	DenseSubMatrix<Number> &J_zc_u = ctxt.get_elem_jacobian(zc_var, u_var);
	DenseSubMatrix<Number> &J_zc_v = ctxt.get_elem_jacobian(zc_var, v_var);
	DenseSubMatrix<Number> &J_zc_c = ctxt.get_elem_jacobian(zc_var, c_var); 
	DenseSubMatrix<Number> &J_zc_auxu = ctxt.get_elem_jacobian(zc_var, aux_u_var);
	DenseSubMatrix<Number> &J_zc_auxv = ctxt.get_elem_jacobian(zc_var, aux_v_var);
	DenseSubMatrix<Number> &J_zc_auxc = ctxt.get_elem_jacobian(zc_var, aux_c_var);
	DenseSubMatrix<Number> &J_zc_auxfc = ctxt.get_elem_jacobian(zc_var, aux_fc_var);
	
	DenseSubMatrix<Number> &J_fc_auxfc = ctxt.get_elem_jacobian(fc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_fc_auxzc = ctxt.get_elem_jacobian(fc_var, aux_zc_var);
	//
	DenseSubMatrix<Number> &J_auxu_u = ctxt.get_elem_jacobian(aux_u_var, u_var);
	DenseSubMatrix<Number> &J_auxu_v = ctxt.get_elem_jacobian(aux_u_var, v_var);
	DenseSubMatrix<Number> &J_auxu_zu = ctxt.get_elem_jacobian(aux_u_var, zu_var);
	DenseSubMatrix<Number> &J_auxu_zp = ctxt.get_elem_jacobian(aux_u_var, zp_var);
	DenseSubMatrix<Number> &J_auxu_zc = ctxt.get_elem_jacobian(aux_u_var, zc_var);
	DenseSubMatrix<Number> &J_auxu_c = ctxt.get_elem_jacobian(aux_u_var, c_var);
	
	DenseSubMatrix<Number> &J_auxv_u = ctxt.get_elem_jacobian(aux_v_var, u_var);
	DenseSubMatrix<Number> &J_auxv_v = ctxt.get_elem_jacobian(aux_v_var, v_var);
	DenseSubMatrix<Number> &J_auxv_zv = ctxt.get_elem_jacobian(aux_v_var, zv_var);
	DenseSubMatrix<Number> &J_auxv_zp = ctxt.get_elem_jacobian(aux_v_var, zp_var);
	DenseSubMatrix<Number> &J_auxv_zc = ctxt.get_elem_jacobian(aux_v_var, zc_var);
	DenseSubMatrix<Number> &J_auxv_c = ctxt.get_elem_jacobian(aux_v_var, c_var);
	
	DenseSubMatrix<Number> &J_auxp_zu = ctxt.get_elem_jacobian(aux_p_var, zu_var);
	DenseSubMatrix<Number> &J_auxp_zv = ctxt.get_elem_jacobian(aux_p_var, zv_var);
	
	DenseSubMatrix<Number> &J_auxc_zc = ctxt.get_elem_jacobian(aux_c_var, zc_var);
	DenseSubMatrix<Number> &J_auxc_u = ctxt.get_elem_jacobian(aux_c_var, u_var);
	DenseSubMatrix<Number> &J_auxc_v = ctxt.get_elem_jacobian(aux_c_var, v_var);
	DenseSubMatrix<Number> &J_auxc_c = ctxt.get_elem_jacobian(aux_c_var, c_var);
	
	DenseSubMatrix<Number> &J_auxzu_u = ctxt.get_elem_jacobian(aux_zu_var, u_var);
	DenseSubMatrix<Number> &J_auxzu_v = ctxt.get_elem_jacobian(aux_zu_var, v_var);
	DenseSubMatrix<Number> &J_auxzu_p = ctxt.get_elem_jacobian(aux_zu_var, p_var);

	DenseSubMatrix<Number> &J_auxzv_u = ctxt.get_elem_jacobian(aux_zv_var, u_var);
	DenseSubMatrix<Number> &J_auxzv_v = ctxt.get_elem_jacobian(aux_zv_var, v_var);
	DenseSubMatrix<Number> &J_auxzv_p = ctxt.get_elem_jacobian(aux_zv_var, p_var);
	
	DenseSubMatrix<Number> &J_auxzp_u = ctxt.get_elem_jacobian(aux_zp_var, u_var);
	DenseSubMatrix<Number> &J_auxzp_v = ctxt.get_elem_jacobian(aux_zp_var, v_var);
	
	DenseSubMatrix<Number> &J_auxzc_u = ctxt.get_elem_jacobian(aux_zc_var, u_var);
	DenseSubMatrix<Number> &J_auxzc_v = ctxt.get_elem_jacobian(aux_zc_var, v_var);
	DenseSubMatrix<Number> &J_auxzc_c = ctxt.get_elem_jacobian(aux_zc_var, c_var);
	DenseSubMatrix<Number> &J_auxzc_fc = ctxt.get_elem_jacobian(aux_zc_var, fc_var);
	
	DenseSubMatrix<Number> &J_auxfc_zc = ctxt.get_elem_jacobian(aux_fc_var, zc_var);
	DenseSubMatrix<Number> &J_auxfc_fc = ctxt.get_elem_jacobian(aux_fc_var, fc_var);

	DenseSubVector<Number> &Ru = ctxt.get_elem_residual( u_var );
	DenseSubVector<Number> &Rv = ctxt.get_elem_residual( v_var );
	DenseSubVector<Number> &Rp = ctxt.get_elem_residual( p_var );
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rzu = ctxt.get_elem_residual( zu_var );
	DenseSubVector<Number> &Rzv = ctxt.get_elem_residual( zv_var );
	DenseSubVector<Number> &Rzp = ctxt.get_elem_residual( zp_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( fc_var );
	DenseSubVector<Number> &Rauxu = ctxt.get_elem_residual( aux_u_var );
	DenseSubVector<Number> &Rauxv = ctxt.get_elem_residual( aux_v_var );
	DenseSubVector<Number> &Rauxp = ctxt.get_elem_residual( aux_p_var );
	DenseSubVector<Number> &Rauxc = ctxt.get_elem_residual( aux_c_var );
	DenseSubVector<Number> &Rauxzu = ctxt.get_elem_residual( aux_zu_var );
	DenseSubVector<Number> &Rauxzv = ctxt.get_elem_residual( aux_zv_var );
	DenseSubVector<Number> &Rauxzp = ctxt.get_elem_residual( aux_zp_var );
	DenseSubVector<Number> &Rauxzc = ctxt.get_elem_residual( aux_zc_var );
	DenseSubVector<Number> &Rauxfc = ctxt.get_elem_residual( aux_fc_var );

	// Now we will build the element Jacobian and residual.
	// Constructing the residual requires the solution and its
	// gradient from the previous timestep.  This must be
	// calculated at each quadrature point by summing the
	// solution degree-of-freedom values by the appropriate
	// weight functions.
	unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

	for (unsigned int qp=0; qp != n_qpoints; qp++)
	  {
	  	//location of quadrature point
	  	const Real ptx = qpoint[qp](0);
	  	const Real pty = qpoint[qp](1);
	  	
	    // Compute the solution & its gradient at the old Newton iterate
	    Number p = ctxt.interior_value(p_var, qp),
	      u = ctxt.interior_value(u_var, qp),
	      v = ctxt.interior_value(v_var, qp),
	      c = ctxt.interior_value(c_var, qp),
	      zu = ctxt.interior_value(zu_var, qp),
	      zv = ctxt.interior_value(zv_var, qp),
	      zp = ctxt.interior_value(zp_var, qp),
	      zc = ctxt.interior_value(zc_var, qp),
	      fc = ctxt.interior_value(fc_var, qp),
	      auxu = ctxt.interior_value(aux_u_var, qp),
	      auxv = ctxt.interior_value(aux_v_var, qp),
	      auxp = ctxt.interior_value(aux_p_var, qp),
	      auxc = ctxt.interior_value(aux_c_var, qp),
	      auxzu = ctxt.interior_value(aux_zu_var, qp),
	      auxzv = ctxt.interior_value(aux_zv_var, qp),
	      auxzp = ctxt.interior_value(aux_zp_var, qp),
	      auxzc = ctxt.interior_value(aux_zc_var, qp),
	      auxfc = ctxt.interior_value(aux_fc_var, qp);
	    Gradient grad_u = ctxt.interior_gradient(u_var, qp),
	      grad_v = ctxt.interior_gradient(v_var, qp),
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zu = ctxt.interior_gradient(zu_var, qp),
	      grad_zv = ctxt.interior_gradient(zv_var, qp),
	      grad_zc = ctxt.interior_gradient(zc_var, qp),
	      grad_fc = ctxt.interior_gradient(fc_var, qp),
	      grad_auxu = ctxt.interior_gradient(aux_u_var, qp),
	      grad_auxv = ctxt.interior_gradient(aux_v_var, qp),
	      grad_auxp = ctxt.interior_gradient(aux_p_var, qp),
	      grad_auxc = ctxt.interior_gradient(aux_c_var, qp),
	      grad_auxzu = ctxt.interior_gradient(aux_zu_var, qp),
	      grad_auxzv = ctxt.interior_gradient(aux_zv_var, qp),
	      grad_auxzp = ctxt.interior_gradient(aux_zp_var, qp),
	      grad_auxzc = ctxt.interior_gradient(aux_zc_var, qp),
	      grad_auxfc = ctxt.interior_gradient(aux_fc_var, qp);

	    // Definitions for convenience.  It is sometimes simpler to do a
	    // dot product if you have the full vector at your disposal.
	    NumberVectorValue U     (u,     v);

	    const Number u_x = grad_u(0); const Number u_y = grad_u(1);
	    const Number v_x = grad_v(0); const Number v_y = grad_v(1);
	    const Number c_x = grad_c(0); const Number c_y = grad_c(1);
	    const Number zc_x = grad_zc(0); const Number zc_y = grad_zc(1);
	    const Number zu_x = grad_zu(0); const Number zu_y = grad_zu(1);
	    const Number zv_x = grad_zv(0); const Number zv_y = grad_zv(1);
	   	const Number auxu_x = grad_auxu(0); const Number auxu_y = grad_auxu(1);
	    const Number auxv_x = grad_auxv(0); const Number auxv_y = grad_auxv(1);
	    const Number auxc_x = grad_auxc(0); const Number auxc_y = grad_auxc(1);
	    const Number auxzc_x = grad_auxzc(0); const Number auxzc_y = grad_auxzc(1);
	    const Number auxzu_x = grad_auxzu(0); const Number auxzu_y = grad_auxzu(1);
	    const Number auxzv_x = grad_auxzv(0); const Number auxzv_y = grad_auxzv(1);

	    //things in velocity's family
	    for (unsigned int i=0; i != n_u_dofs; i++){ 
	    
	    	//recovering L'
	    	Rauxzu(i) += JxW[qp]*(params[0]*U*grad_u*phi[i][qp] - p*dphi[i][qp](0) + params[1]*grad_u*dphi[i][qp]);
	    	Rauxzv(i) += JxW[qp]*(params[0]*U*grad_v*phi[i][qp] - p*dphi[i][qp](1) + params[1]*grad_v*dphi[i][qp]);
	    	Rauxu(i) += JxW[qp]*(params[1]*grad_zu*dphi[i][qp] - zp*dphi[i][qp](0) - zc*c_x*phi[i][qp]
  														+ params[0]*(-U*grad_zu - v_y*zu)*phi[i][qp]);
	    	Rauxv(i) += JxW[qp]*(params[1]*grad_zv*dphi[i][qp] - zp*dphi[i][qp](1) - zc*c_y*phi[i][qp]
	      										+ params[0]*(-U*grad_zv - u_x*zv)*phi[i][qp]);
	    	
	    	//other parts in M'		
	     	Ru(i) += JxW[qp]*(params[0]*(-U*grad_auxzu - v_y*auxzu + auxzv*v_x - auxu*zu_x + zv*auxv_x)*phi[i][qp] 
	     											+ params[1]*grad_auxzu*dphi[i][qp]
	     											- auxzp*dphi[i][qp](0) - auxzc*c_x*phi[i][qp] - zc*auxc_x*phi[i][qp]);
	     	Rv(i) += JxW[qp]*(params[0]*(-U*grad_auxzv - u_x*auxzv + auxzu*u_y - auxv*zv_y + zu*auxu_y)*phi[i][qp]
	     											+ params[1]*grad_auxzv*dphi[i][qp]
	     											- auxzp*dphi[i][qp](1) - auxzc*c_y*phi[i][qp] - zc*auxc_y*phi[i][qp]);
	     	Rzu(i) += JxW[qp]*(params[0]*(U*grad_auxu + u_x*auxu)*phi[i][qp]
	     											+ params[1]*grad_auxu*dphi[i][qp] - auxp*dphi[i][qp](0));
	     	Rzv(i) += JxW[qp]*(params[0]*(U*grad_auxv + v_y*auxv)*phi[i][qp]
	     											+ params[1]*grad_auxv*dphi[i][qp] - auxp*dphi[i][qp](1));

	      if (request_jacobian && ctxt.elem_solution_derivative){
	        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

	        for (unsigned int j=0; j != n_u_dofs; j++){
	        	J_auxzu_u(i,j) += JxW[qp]*(params[1]*dphi[j][qp]*dphi[i][qp]
	        										+ params[0]*U*dphi[j][qp]*phi[i][qp] + params[0]*phi[j][qp]*u_x*phi[i][qp]);
	        	J_auxzv_v(i,j) += JxW[qp]*(params[1]*dphi[j][qp]*dphi[i][qp]
	        										+ params[0]*U*dphi[j][qp]*phi[i][qp] + params[0]*phi[j][qp]*v_y*phi[i][qp]);
	        	J_auxzu_v(i,j) += JxW[qp]*(params[0]*phi[j][qp]*u_y*phi[i][qp]);
	        	J_auxzv_u(i,j) += JxW[qp]*(params[0]*phi[j][qp]*v_x*phi[i][qp]);
	        	
	        	J_auxu_u(i,j) += JxW[qp]*(params[0]*(-phi[j][qp]*zu_x)*phi[i][qp]);
	        	J_auxu_v(i,j) += JxW[qp]*(params[0]*(-phi[j][qp]*zu_y - dphi[j][qp](1)*zu)*phi[i][qp]);
	        	J_auxv_u(i,j) += JxW[qp]*(params[0]*(-phi[j][qp]*zv_x - dphi[j][qp](0)*zv)*phi[i][qp]);
	        	J_auxv_v(i,j) += JxW[qp]*(params[0]*(-phi[j][qp]*zv_y)*phi[i][qp]);
	        	J_auxu_zu(i,j) += JxW[qp]*(params[1]*dphi[j][qp]*dphi[i][qp]
	        										+ params[0]*(-U*dphi[j][qp] - v_y*phi[j][qp])*phi[i][qp]);
	        	J_auxv_zv(i,j) += JxW[qp]*(params[1]*dphi[j][qp]*dphi[i][qp]
	        										+ params[0]*(-U*dphi[j][qp] - u_x*phi[j][qp])*phi[i][qp]);
	     											
	     			J_u_u(i,j) += JxW[qp]*params[0]*(-phi[j][qp]*auxzu_x)*phi[i][qp];
	     			J_u_v(i,j) += JxW[qp]*params[0]*(-phi[j][qp]*auxzu_y - dphi[j][qp](1)*auxzu + dphi[j][qp](0)*auxzv)*phi[i][qp];
	     			J_u_zu(i,j) += JxW[qp]*params[0]*(-auxu*dphi[j][qp](0))*phi[i][qp];
	     			J_u_zv(i,j) += JxW[qp]*params[0]*auxv_x*phi[j][qp]*phi[i][qp];
	     			J_u_auxzu(i,j) += JxW[qp]*(params[0]*(-U*dphi[j][qp] - v_y*phi[j][qp])*phi[i][qp] 
	     														+ params[1]*dphi[j][qp]*dphi[i][qp]);
	     			J_u_auxzv(i,j) += JxW[qp]*params[0]*(phi[j][qp]*v_x)*phi[i][qp];
	     			J_u_auxu(i,j) += JxW[qp]*params[0]*(-phi[j][qp]*zu_x)*phi[i][qp];
	     			J_u_auxv(i,j) += JxW[qp]*params[0]*(zv*dphi[j][qp](0))*phi[i][qp];
	     			
	     			J_v_u(i,j) += JxW[qp]*params[0]*(-phi[j][qp]*auxzv_x - dphi[j][qp](0)*auxzv + dphi[j][qp](1)*auxzu)*phi[i][qp];
	     			J_v_v(i,j) += JxW[qp]*params[0]*(-phi[j][qp]*auxzv_y)*phi[i][qp];
	     			J_v_zu(i,j) += JxW[qp]*params[0]*auxu_y*phi[j][qp]*phi[i][qp];
	     			J_v_zv(i,j) += JxW[qp]*params[0]*(-auxv*dphi[j][qp](1))*phi[i][qp];
	     			J_v_auxzu(i,j) += JxW[qp]*params[0]*(u_y*phi[j][qp])*phi[i][qp];
	     			J_v_auxzv(i,j) += JxW[qp]*(params[0]*(-U*dphi[j][qp] - u_x*phi[j][qp])*phi[i][qp]
	     														+ params[1]*dphi[j][qp]*dphi[i][qp]);
	     			J_v_auxu(i,j) += JxW[qp]*params[0]*zu*dphi[j][qp](1)*phi[i][qp];
	     			J_v_auxv(i,j) += JxW[qp]*params[0]*(-zv_y*phi[j][qp])*phi[i][qp];
	     											
	     			J_zu_u(i,j) += JxW[qp]*params[0]*(phi[j][qp]*auxu_x + dphi[j][qp](0)*auxu)*phi[i][qp];
	     			J_zu_v(i,j) += JxW[qp]*params[0]*phi[j][qp]*auxu_y*phi[i][qp];
	     			J_zu_auxu(i,j) += JxW[qp]*(params[0]*(U*dphi[j][qp] + u_x*phi[j][qp])*phi[i][qp]
	     															+ params[1]*dphi[j][qp]*dphi[i][qp]);
	     											
	     			J_zv_u(i,j) += JxW[qp]*params[0]*phi[j][qp]*auxv_x*phi[i][qp];
	     			J_zv_v(i,j) += JxW[qp]*params[0]*(phi[j][qp]*auxv_y + dphi[j][qp](1)*auxv)*phi[i][qp];
	     			J_zv_auxv(i,j) += JxW[qp]*(params[0]*(U*dphi[j][qp] + v_y*phi[j][qp])*phi[i][qp]
	     															+ params[1]*dphi[j][qp]*dphi[i][qp]);								
	     			
	        }

	        for (unsigned int j=0; j != n_p_dofs; j++){
	          J_auxzu_p(i,j) += JxW[qp]*(-psi[j][qp])*dphi[i][qp](0);
	          J_auxzv_p(i,j) += JxW[qp]*(-psi[j][qp])*dphi[i][qp](1);
	          
	          J_auxu_zp(i,j) += JxW[qp]*(-psi[j][qp]*dphi[i][qp](0));
	          J_auxu_zc(i,j) += JxW[qp]*(-psi[j][qp]*c_x*phi[i][qp]);
	          J_auxu_c(i,j) += JxW[qp]*(-zc*dpsi[j][qp](0)*phi[i][qp]);
	          J_auxv_zp(i,j) += JxW[qp]*(-psi[j][qp]*dphi[i][qp](1));
	          J_auxv_zc(i,j) += JxW[qp]*(-psi[j][qp]*c_y*phi[i][qp]);
	          J_auxv_c(i,j) += JxW[qp]*(-zc*dpsi[j][qp](1)*phi[i][qp]);
	         
	          J_u_c(i,j) += -JxW[qp]*auxzc*dpsi[j][qp](0)*phi[i][qp];
	          J_u_zc(i,j) += -JxW[qp]*auxc_x*psi[j][qp]*phi[i][qp];
	          J_u_auxzc(i,j) += -JxW[qp]*c_x*psi[j][qp]*phi[i][qp];
	          J_u_auxzp(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);
	          J_u_auxc(i,j) += -JxW[qp]*zc*dpsi[j][qp](0)*phi[i][qp];
								
						J_v_c(i,j) += -JxW[qp]*auxzc*dpsi[j][qp](1)*phi[i][qp];
						J_v_zc(i,j) += -JxW[qp]*auxc_y*psi[j][qp]*phi[i][qp];
						J_v_auxzc(i,j) += -JxW[qp]*psi[j][qp]*c_y*phi[i][qp];
						J_v_auxzp(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);
						J_v_auxc(i,j) += -JxW[qp]*zc*dpsi[j][qp](1)*phi[i][qp];

						J_zu_auxp(i,j) += JxW[qp]*(-psi[j][qp])*dphi[i][qp](0);
						J_zv_auxp(i,j) += JxW[qp]*(-psi[j][qp])*dphi[i][qp](1);
	        }
	      }
	    }
	    
	    //things in pressure's family
	    for (unsigned int i=0; i != n_p_dofs; i++){ 
	    	//recovering L'
	      Rauxzp(i) += JxW[qp]*(-u_x*psi[i][qp] - v_y*psi[i][qp]);
	      Rauxzc(i) += JxW[qp]*(-grad_c*dpsi[i][qp] - U*grad_c*psi[i][qp] + fc*psi[i][qp]);
	      Rauxp(i) += JxW[qp]*(-zu_x*psi[i][qp] - zv_y*psi[i][qp]);
	      Rauxc(i) += JxW[qp]*(-grad_zc*dpsi[i][qp] + (U*grad_zc + zc*(u_x + v_y))*psi[i][qp]);
	      if(regtype == 0)
	      	Rauxfc(i) += JxW[qp]*(beta*fc*psi[i][qp] + zc*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rauxfc(i) += JxW[qp]*(beta*grad_fc*dpsi[i][qp] + zc*psi[i][qp]);
	     		
	     	//other parts in M'
	     	Rp(i) += JxW[qp]*(auxzu*dpsi[i][qp](0) + auxzv*dpsi[i][qp](1));
	     	Rc(i) += JxW[qp]*(-grad_auxzc*dpsi[i][qp] + (U*grad_auxzc + auxzc*(u_x + v_y))*psi[i][qp]
     									+ (zc*auxu_x + zc_x*auxu + zc*auxv_y + zc_y*auxv)*psi[i][qp]
     									+ auxc*psi[i][qp]);
     		if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125) //is this correct?
     			Rc(i) += JxW[qp];
	     	Rzp(i) += JxW[qp]*(auxu*dpsi[i][qp](0) + auxv*dpsi[i][qp](1));
	     	Rzc(i) += JxW[qp]*((-auxu*c_x - auxv*c_y + auxfc - U*grad_auxc)*psi[i][qp]
	     								- grad_auxc*dpsi[i][qp]);
	     	if(regtype == 0)
	     		Rfc(i) += JxW[qp]*((auxzc + beta*auxfc)*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rfc(i) += JxW[qp]*(auxzc*psi[i][qp] + beta*grad_auxfc*dpsi[i][qp]);
	      
	      if (request_jacobian && ctxt.elem_solution_derivative){
	        for(unsigned int j=0; j != n_u_dofs; j++){
	          J_auxzp_u(i,j) += JxW[qp]*(-dphi[j][qp](0)*psi[i][qp]);
	        	J_auxzp_v(i,j) += JxW[qp]*(-dphi[j][qp](1)*psi[i][qp]);
	        	J_auxzc_u(i,j) += JxW[qp]*(-phi[j][qp]*c_x*psi[i][qp]);
	        	J_auxzc_v(i,j) += JxW[qp]*(-phi[j][qp]*c_y*psi[i][qp]);
	        	
	        	J_auxp_zu(i,j) += JxW[qp]*(-dphi[j][qp](0)*psi[i][qp]);
	        	J_auxp_zv(i,j) += JxW[qp]*(-dphi[j][qp](1)*psi[i][qp]);
	        	J_auxc_u(i,j) += JxW[qp]*(phi[j][qp]*zc_x*psi[i][qp] + zc*dphi[j][qp](0)*psi[i][qp]);
	        	J_auxc_v(i,j) += JxW[qp]*(phi[j][qp]*zc_y*psi[i][qp] + zc*dphi[j][qp](1)*psi[i][qp]);
	        	
	        	J_p_auxzu(i,j) += JxW[qp]*(phi[j][qp]*dpsi[i][qp](0));
	        	J_p_auxzv(i,j) += JxW[qp]*(phi[j][qp]*dpsi[i][qp](1));
	        	
	        	J_c_u(i,j) += JxW[qp]*(phi[j][qp]*auxzc_x + auxzc*dphi[j][qp](0))*psi[i][qp];
	        	J_c_v(i,j) += JxW[qp]*(phi[j][qp]*auxzc_y + auxzc*dphi[j][qp](1))*psi[i][qp];
	        	J_c_auxu(i,j) += JxW[qp]*(zc*dphi[j][qp](0) + zc_x*phi[j][qp])*psi[i][qp];
	        	J_c_auxv(i,j) += JxW[qp]*(zc*dphi[j][qp](1) + zc_y*phi[j][qp])*psi[i][qp];
	        	
	        	J_zp_auxu(i,j) += JxW[qp]*phi[j][qp]*dpsi[i][qp](0);
	        	J_zp_auxv(i,j) += JxW[qp]*phi[j][qp]*dpsi[i][qp](1);	
	     								      	
	        	J_zc_u(i,j) += JxW[qp]*(-phi[j][qp]*auxc_x*psi[i][qp]);
	        	J_zc_v(i,j) += JxW[qp]*(-phi[j][qp]*auxc_y*psi[i][qp]);
	        	J_zc_auxu(i,j) += JxW[qp]*(-phi[j][qp]*c_x)*psi[i][qp];
	        	J_zc_auxv(i,j) += JxW[qp]*(-phi[j][qp]*c_y)*psi[i][qp];
	        	
	       	}
	       	
	       	for(unsigned int j=0; j != n_p_dofs; j++){
	       		J_auxzc_c(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] - U*dpsi[j][qp]*psi[i][qp]);
	       		J_auxzc_fc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		
	       		J_auxc_zc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] + (U*dpsi[j][qp] + psi[j][qp]*(u_x + v_y))*psi[i][qp]);
	       		J_auxfc_zc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		if(regtype == 0)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*psi[j][qp]*psi[i][qp]);
	       		else if(regtype == 1)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*dpsi[j][qp]*dpsi[i][qp]);
     				
     				if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)					
							J_c_c(i,j) += 0; //no dependence on c here if QoI is integral of c over subdomain
						J_c_zc(i,j) += JxW[qp]*(psi[j][qp]*(auxu_x + auxv_y) 
																+ dpsi[j][qp](0)*auxu + dpsi[j][qp](1)*auxv)*psi[i][qp];
						J_c_auxzc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp]
																+ (U*dpsi[j][qp] + psi[j][qp]*(u_x + v_y))*psi[i][qp]);
						J_c_auxc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	     													
						J_zc_c(i,j) += JxW[qp]*(-auxu*dpsi[j][qp](0) - auxv*dpsi[j][qp](1))*psi[i][qp];
						J_zc_auxc(i,j) += JxW[qp]*(-U*dpsi[j][qp]*psi[i][qp] - dpsi[j][qp]*dpsi[i][qp]);
						J_zc_auxfc(i,j) += JxW[qp]*psi[j][qp]*psi[i][qp];
	     		
						if(regtype == 0){
			      	J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*psi[j][qp]*psi[i][qp];
	        	}
	        	else if(regtype == 1){
	        		J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*dpsi[j][qp]*dpsi[i][qp];
	        	}
	       	}
	      }
	    }
	  } // end of the quadrature point qp-loop
	  
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	if(ctxt.get_elem().contains_point(data_point)){
	  		Number cpred = ctxt.point_value(c_var, data_point);
	  		Number cstar = datavals[dnum];
	  		
	  		unsigned int dim = get_mesh().mesh_dimension();
		    FEType fe_type = p_elem_fe->get_fe_type();
		    
		    //go between physical and reference element
		    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		    
        std::vector<Real> point_phi(n_p_dofs);
      	for (unsigned int i=0; i != n_p_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_p_dofs; i++){
  	  		Rauxc(i) += (cpred - cstar)*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_p_dofs; j++)
							J_auxc_c(i,j) += point_phi[j]*point_phi[i] ;
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}


void NavStokesConvDiffSys::postprocess()
{
  //reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);

}

