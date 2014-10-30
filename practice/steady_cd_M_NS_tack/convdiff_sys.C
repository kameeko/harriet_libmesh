#include "libmesh/getpot.h"

#include "convdiff_sys.h"

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
void ConvDiffSys::init_data (){
	const unsigned int dim = this->get_mesh().mesh_dimension();

	//polynomial order and finite element type for pressure variable
	unsigned int pressure_p = 1;
	GetPot infile("convdiff.in");
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

	//polynomial order and finite element type for concentration variable
	unsigned int conc_p = 1;
	                                                      
	c_var = this->add_variable("c", static_cast<Order>(conc_p), fefamily);   
	zc_var = this->add_variable("zc", static_cast<Order>(conc_p), fefamily); 
	
	//source parameter in same space as the thing it spews...
	fc_var = this->add_variable("fc", static_cast<Order>(conc_p), fefamily);  
	
	//auxillary variables 
	aux_c_var = this->add_variable("aux_c", static_cast<Order>(conc_p), fefamily); 
	aux_zc_var = this->add_variable("aux_zc", static_cast<Order>(conc_p), fefamily); 
	aux_fc_var = this->add_variable("aux_fc", static_cast<Order>(conc_p), fefamily);  

	//regularization
	beta = infile("beta",0.1);
	regtype = infile("regularization_option",1);

	//indicate variables that change in time
	this->time_evolving(c_var);
	this->time_evolving(zc_var);
	this->time_evolving(fc_var);
	this->time_evolving(aux_c_var);
	this->time_evolving(aux_zc_var);
	this->time_evolving(aux_fc_var);
	this->time_evolving(u_var);
	this->time_evolving(v_var);
	this->time_evolving(p_var);

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

	std::vector<unsigned int> c_only(1, c_var);
	std::vector<unsigned int> not_c_or_f(1, zc_var);
		not_c_or_f.push_back(aux_c_var); not_c_or_f.push_back(aux_zc_var);
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
	  
	//c=0 on boundary, cuz I feel like it...
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, c_only, &zero));
	
	//corresponding BCs for adjoints and auxillary variables
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, not_c_or_f, &zero));
	
	if(regtype == 1){
		std::vector<unsigned int> fc_only(1, fc_var); fc_only.push_back(aux_fc_var);
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, fc_only, &zero));
	}
	  
	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

// Context initialization
void ConvDiffSys::init_context(DiffContext &context){
	FEMContext &ctxt = cast_ref<FEMContext&>(context);

	FEBase* c_elem_fe = NULL;

  ctxt.get_element_fe(c_var, c_elem_fe);
  c_elem_fe->get_JxW();
  c_elem_fe->get_phi();
  c_elem_fe->get_dphi();

  FEBase* c_side_fe = NULL;
  ctxt.get_side_fe(c_var, c_side_fe);

  c_side_fe->get_JxW();
  c_side_fe->get_phi();
  c_side_fe->get_dphi();
  
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
}

// Element residual and jacobian calculations
// Time dependent parts
bool ConvDiffSys::element_time_derivative (bool request_jacobian, DiffContext& context){

	FEMContext &ctxt = cast_ref<FEMContext&>(context);

	//some cell-specific stuff
	FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( c_var, c_elem_fe );
  FEBase* u_elem_fe = NULL; 
  ctxt.get_element_fe( u_var, u_elem_fe );

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = c_elem_fe->get_JxW();

	//for velocities, at interior quadrature points
	const std::vector<std::vector<Real> >& phi = u_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi = u_elem_fe->get_dphi();

	//for pressure and concentration, at interior quadrature points
	const std::vector<std::vector<Real> >& psi = c_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dpsi = c_elem_fe->get_dphi();

	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();
	const unsigned int n_u_dofs = ctxt.get_dof_indices( u_var ).size();
	libmesh_assert_equal_to (n_u_dofs, ctxt.get_dof_indices( v_var ).size());

	// The subvectors and submatrices we need to fill:
	const unsigned int dim = this->get_mesh().mesh_dimension();

	DenseSubMatrix<Number> &J_c_auxzc = ctxt.get_elem_jacobian(c_var, aux_zc_var);
	DenseSubMatrix<Number> &J_c_auxc = ctxt.get_elem_jacobian(c_var, aux_c_var);
	DenseSubMatrix<Number> &J_c_c = ctxt.get_elem_jacobian(c_var, c_var);
	DenseSubMatrix<Number> &J_c_u = ctxt.get_elem_jacobian(c_var, u_var);
	DenseSubMatrix<Number> &J_c_v = ctxt.get_elem_jacobian(c_var, v_var);
	
	DenseSubMatrix<Number> &J_zc_auxc = ctxt.get_elem_jacobian(zc_var, aux_c_var);
	DenseSubMatrix<Number> &J_zc_auxfc = ctxt.get_elem_jacobian(zc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_zc_u = ctxt.get_elem_jacobian(zc_var, u_var);
	DenseSubMatrix<Number> &J_zc_v = ctxt.get_elem_jacobian(zc_var, v_var);
	
	DenseSubMatrix<Number> &J_fc_auxfc = ctxt.get_elem_jacobian(fc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_fc_auxzc = ctxt.get_elem_jacobian(fc_var, aux_zc_var);
	
	DenseSubMatrix<Number> &J_auxc_zc = ctxt.get_elem_jacobian(aux_c_var, zc_var);
	DenseSubMatrix<Number> &J_auxc_c = ctxt.get_elem_jacobian(aux_c_var, c_var);
	DenseSubMatrix<Number> &J_auxc_u = ctxt.get_elem_jacobian(aux_c_var, u_var);
	DenseSubMatrix<Number> &J_auxc_v = ctxt.get_elem_jacobian(aux_c_var, v_var);
	
	DenseSubMatrix<Number> &J_auxzc_c = ctxt.get_elem_jacobian(aux_zc_var, c_var);
	DenseSubMatrix<Number> &J_auxzc_fc = ctxt.get_elem_jacobian(aux_zc_var, fc_var);
	DenseSubMatrix<Number> &J_auxzc_u = ctxt.get_elem_jacobian(aux_zc_var, u_var);
	DenseSubMatrix<Number> &J_auxzc_v = ctxt.get_elem_jacobian(aux_zc_var, v_var);
	
	DenseSubMatrix<Number> &J_auxfc_zc = ctxt.get_elem_jacobian(aux_fc_var, zc_var);
	DenseSubMatrix<Number> &J_auxfc_fc = ctxt.get_elem_jacobian(aux_fc_var, fc_var);
	
	DenseSubMatrix<Number> &J_NSu_u = ctxt.get_elem_jacobian(u_var, u_var);
	DenseSubMatrix<Number> &J_NSu_v = ctxt.get_elem_jacobian(u_var, v_var);
	DenseSubMatrix<Number> &J_NSu_p = ctxt.get_elem_jacobian(u_var, p_var);
	
	DenseSubMatrix<Number> &J_NSv_u = ctxt.get_elem_jacobian(v_var, u_var);
	DenseSubMatrix<Number> &J_NSv_v = ctxt.get_elem_jacobian(v_var, v_var);
	DenseSubMatrix<Number> &J_NSv_p = ctxt.get_elem_jacobian(v_var, p_var);
	
	DenseSubMatrix<Number> &J_NSp_u = ctxt.get_elem_jacobian(p_var, u_var);
	DenseSubMatrix<Number> &J_NSp_v = ctxt.get_elem_jacobian(p_var, v_var);

	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( fc_var );
	DenseSubVector<Number> &Rauxc = ctxt.get_elem_residual( aux_c_var );;
	DenseSubVector<Number> &Rauxzc = ctxt.get_elem_residual( aux_zc_var );
	DenseSubVector<Number> &Rauxfc = ctxt.get_elem_residual( aux_fc_var );
	DenseSubVector<Number> &RNSu = ctxt.get_elem_residual( u_var );
	DenseSubVector<Number> &RNSv= ctxt.get_elem_residual( v_var );
	DenseSubVector<Number> &RNSp = ctxt.get_elem_residual( p_var );

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
	    Number 
	      c = ctxt.interior_value(c_var, qp),
	      zc = ctxt.interior_value(zc_var, qp),
	      fc = ctxt.interior_value(fc_var, qp),
	      auxc = ctxt.interior_value(aux_c_var, qp),
	      auxzc = ctxt.interior_value(aux_zc_var, qp),
	      auxfc = ctxt.interior_value(aux_fc_var, qp),
	      p = ctxt.interior_value(p_var, qp),
	      u = ctxt.interior_value(u_var, qp),
	      v = ctxt.interior_value(v_var, qp);
	    Gradient 
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zc = ctxt.interior_gradient(zc_var, qp),
	      grad_fc = ctxt.interior_gradient(fc_var, qp),
	      grad_auxc = ctxt.interior_gradient(aux_c_var, qp),
	      grad_auxzc = ctxt.interior_gradient(aux_zc_var, qp),
	      grad_auxfc = ctxt.interior_gradient(aux_fc_var, qp),
	      grad_u = ctxt.interior_gradient(u_var, qp),
	      grad_v = ctxt.interior_gradient(v_var, qp),
	      grad_p = ctxt.interior_gradient(p_var, qp);

	    // Definitions for convenience.  It is sometimes simpler to do a
	    // dot product if you have the full vector at your disposal.
	    NumberVectorValue U     (u,     v);
	    
	    const Number c_x = grad_c(0); const Number c_y = grad_c(1);
	    const Number zc_x = grad_zc(0); const Number zc_y = grad_zc(1);
	    const Number auxc_x = grad_auxc(0); const Number auxc_y = grad_auxc(1);
	    const Number auxzc_x = grad_auxzc(0); const Number auxzc_y = grad_auxzc(1);
	    const Number u_x = grad_u(0); const Number u_y = grad_u(1);
	    const Number v_x = grad_v(0); const Number v_y = grad_v(1);

	    for (unsigned int i=0; i != n_c_dofs; i++){ 
	      Rauxc(i) += JxW[qp]*(-grad_zc*dpsi[i][qp] + U*grad_zc*psi[i][qp]);
	      Rauxzc(i) += JxW[qp]*(-grad_c*dpsi[i][qp] - U*grad_c*psi[i][qp] + fc*psi[i][qp]);
	      if(regtype == 0)
	      	Rauxfc(i) += JxW[qp]*(beta*fc*psi[i][qp] + zc*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rauxfc(i) += JxW[qp]*(beta*grad_fc*dpsi[i][qp] + zc*psi[i][qp]);
	     		
	      Rc(i) += JxW[qp]*(-grad_auxzc*dpsi[i][qp] + U*grad_auxzc*psi[i][qp] + auxc*psi[i][qp]);
	      if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125) //is this correct?
     			Rc(i) += JxW[qp]*psi[i][qp]; //Rc(i) += JxW[qp]?
	      Rzc(i) += JxW[qp]*(-grad_auxc*dpsi[i][qp] - U*grad_auxc*psi[i][qp] + auxfc*psi[i][qp]);
	     	if(regtype == 0)
   				Rfc(i) += JxW[qp]*((auxzc + beta*auxfc)*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rfc(i) += JxW[qp]*(auxzc*psi[i][qp] + beta*grad_auxfc*dpsi[i][qp]);
	     		
	     	RNSp(i) += -JxW[qp]*(u_x*psi[i][qp] + v_y*psi[i][qp]);

	      if (request_jacobian && ctxt.elem_solution_derivative){
	        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

	        for (unsigned int j=0; j != n_c_dofs; j++){
	        
        		J_c_auxzc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp]);
						J_c_auxc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
     				if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)					
							J_c_c(i,j) += 0; //no dependence on c here if QoI is integral of c over subdomain
	
						J_zc_auxc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] - U*dpsi[j][qp]*psi[i][qp]);
						J_zc_auxfc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	
						if(regtype == 0){
			      	J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*psi[j][qp]*psi[i][qp];
	        	}
	        	else if(regtype == 1){
	        		J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*dpsi[j][qp]*dpsi[i][qp];
	        	}
	    
						J_auxc_zc(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp]);

						J_auxzc_c(i,j) += JxW[qp]*(-dpsi[j][qp]*dpsi[i][qp] - U*dpsi[j][qp]*psi[i][qp]);
						J_auxzc_fc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	
	       		J_auxfc_zc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		if(regtype == 0)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*psi[j][qp]*psi[i][qp]);
	       		else if(regtype == 1)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*dpsi[j][qp]*dpsi[i][qp]);

	        }
	        for (unsigned int j=0; j != n_u_dofs; j++){
	        
	        	J_c_u(i,j) += JxW[qp]*(phi[j][qp]*auxzc_x*psi[i][qp]);
	        	J_c_v(i,j) += JxW[qp]*(phi[j][qp]*auxzc_y*psi[i][qp]);
	        	
	        	J_zc_u(i,j) += JxW[qp]*(-phi[j][qp]*auxc_x*psi[i][qp]);
	        	J_zc_v(i,j) += JxW[qp]*(-phi[j][qp]*auxc_y*psi[i][qp]);
	        	
	        	J_auxc_u(i,j) += JxW[qp]*(phi[j][qp]*zc_x*psi[i][qp]);
	        	J_auxc_v(i,j) += JxW[qp]*(phi[j][qp]*zc_y*psi[i][qp]);
	        	
	        	J_auxzc_u(i,j) += JxW[qp]*(-phi[j][qp]*c_x*psi[i][qp]);
	        	J_auxzc_v(i,j) += JxW[qp]*(-phi[j][qp]*c_y*psi[i][qp]);
	        	
	        	J_NSp_u(i,j) += -JxW[qp]*(dphi[j][qp](0)*psi[i][qp]);
	        	J_NSp_v(i,j) += -JxW[qp]*(dphi[j][qp](1)*psi[i][qp]);
	        }
	      }
	    }
	    for (unsigned int i=0; i != n_u_dofs; i++){
	    	RNSu(i) += -JxW[qp]*(-params[0]*U*grad_u*phi[i][qp] + p*dphi[i][qp](0) - params[1]*(grad_u*dphi[i][qp]));
	    	RNSv(i) += -JxW[qp]*(-params[0]*U*grad_v*phi[i][qp] + p*dphi[i][qp](1) - params[1]*(grad_v*dphi[i][qp]));
	    	if (request_jacobian && ctxt.elem_solution_derivative){
	        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

	        for (unsigned int j=0; j != n_c_dofs; j++){
						J_NSu_p(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](0);
	          J_NSv_p(i,j) += -JxW[qp]*psi[j][qp]*dphi[i][qp](1);
	        }
	        for (unsigned int j=0; j != n_u_dofs; j++){
	        	J_NSu_u(i,j) += -JxW[qp]*(-params[1]*dphi[j][qp]*dphi[i][qp]
	        										- params[0]*U*dphi[j][qp]*phi[i][qp] - params[0]*phi[j][qp]*u_x*phi[i][qp]);
	        	J_NSv_v(i,j) += -JxW[qp]*(-params[1]*dphi[j][qp]*dphi[i][qp]
	        										- params[0]*U*dphi[j][qp]*phi[i][qp] - params[0]*phi[j][qp]*v_y*phi[i][qp]);
	        	J_NSu_v(i,j) += -JxW[qp]*(-params[0]*phi[j][qp]*u_y*phi[i][qp]);
	        	J_NSv_u(i,j) += -JxW[qp]*(-params[0]*phi[j][qp]*v_x*phi[i][qp]);
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
		    FEType fe_type = c_elem_fe->get_fe_type();
		    
		    //go between physical and reference element
		    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		    
        std::vector<Real> point_phi(n_c_dofs);
      	for (unsigned int i=0; i != n_c_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_c_dofs; i++){
  	  		Rauxc(i) += (cpred - cstar)*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_c_dofs; j++)
							J_auxc_c(i,j) += point_phi[j]*point_phi[i] ;
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}


void ConvDiffSys::postprocess()
{
  //reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);

}

bool ConvDiffSys::side_constraint (bool request_jacobian,
                                    DiffContext &context)
{
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* p_elem_fe;

  ctxt.get_element_fe( p_var, p_elem_fe );

  // Pin p = 0 at the origin
  const Point zero(0.,0.);

  if( ctxt.get_elem().contains_point(zero))
    {
      // The pressure penalty value.  \f$ \frac{1}{\epsilon} \f$
      const Real penalty = 1.e9;

      DenseSubMatrix<Number> &Jpp = ctxt.get_elem_jacobian( p_var, p_var );
      
      DenseSubVector<Number> &Rp = ctxt.get_elem_residual( p_var );

      const unsigned int n_p_dofs = ctxt.get_dof_indices( p_var ).size();

      Number p = ctxt.point_value(p_var, zero);
      Number p_pin = 0.;

      unsigned int dim = get_mesh().mesh_dimension();
      FEType fe_type = p_elem_fe->get_fe_type();
      Point p_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), zero);

      std::vector<Real> point_phi(n_p_dofs);
      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, p_master);
        }

      for (unsigned int i=0; i != n_p_dofs; i++)
        {
          Rp(i) += penalty * (p - p_pin) * point_phi[i];
          if (request_jacobian && ctxt.get_elem_solution_derivative())
            {
              libmesh_assert_equal_to (ctxt.get_elem_solution_derivative(), 1.0);

              for (unsigned int j=0; j != n_p_dofs; j++){
                Jpp(i,j) += penalty * point_phi[i] * point_phi[j];
              }
            }
        }
    }

  return request_jacobian;
}
