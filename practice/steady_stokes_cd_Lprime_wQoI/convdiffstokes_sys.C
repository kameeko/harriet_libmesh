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
	
	Peclet = infile("Peclet", 1.0);
	parameters.push_back(Peclet);

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
	
	//corresponding BCs on for adjoints
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, z_forDiri, &zero));
	
	if(regtype == 1){
		std::vector<unsigned int> fc_only(1, fc_var);
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, fc_only, &zero));
	}
	  
	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

// Context initialization
void StokesConvDiffSys::init_context(DiffContext &context){
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
bool StokesConvDiffSys::element_time_derivative (bool request_jacobian, DiffContext& context){

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
	DenseSubMatrix<Number> &Juu = ctxt.get_elem_jacobian( u_var, u_var );
	DenseSubMatrix<Number> &Juv = ctxt.get_elem_jacobian( u_var, v_var );
	DenseSubMatrix<Number> &Jup = ctxt.get_elem_jacobian( u_var, p_var );

	DenseSubMatrix<Number> &Jvv = ctxt.get_elem_jacobian( v_var, v_var );
	DenseSubMatrix<Number> &Jvu = ctxt.get_elem_jacobian( v_var, u_var ); 
	DenseSubMatrix<Number> &Jvp = ctxt.get_elem_jacobian( v_var, p_var );
	
	DenseSubMatrix<Number> &Jpu = ctxt.get_elem_jacobian( p_var, u_var );
	DenseSubMatrix<Number> &Jpv = ctxt.get_elem_jacobian( p_var, v_var );
	
	DenseSubMatrix<Number> &Jcc = ctxt.get_elem_jacobian(c_var, c_var);	
	DenseSubMatrix<Number> &Jcu = ctxt.get_elem_jacobian( c_var, u_var );
	DenseSubMatrix<Number> &Jcv = ctxt.get_elem_jacobian( c_var, v_var );
	DenseSubMatrix<Number> &J_c_fc = ctxt.get_elem_jacobian(c_var, fc_var);
	
	DenseSubMatrix<Number> &J_zu_zu = ctxt.get_elem_jacobian(zu_var, zu_var);
	DenseSubMatrix<Number> &J_zu_zp = ctxt.get_elem_jacobian(zu_var, zp_var);
	DenseSubMatrix<Number> &J_zu_zc = ctxt.get_elem_jacobian(zu_var, zc_var);
	DenseSubMatrix<Number> &J_zu_c = ctxt.get_elem_jacobian(zu_var, c_var);
	
	DenseSubMatrix<Number> &J_zv_zv = ctxt.get_elem_jacobian(zv_var, zv_var);
	DenseSubMatrix<Number> &J_zv_zp = ctxt.get_elem_jacobian(zv_var, zp_var);
	DenseSubMatrix<Number> &J_zv_zc = ctxt.get_elem_jacobian(zv_var, zc_var);
	DenseSubMatrix<Number> &J_zv_c = ctxt.get_elem_jacobian(zv_var, c_var);
	
	DenseSubMatrix<Number> &J_zp_zu = ctxt.get_elem_jacobian(zp_var, zu_var);
	DenseSubMatrix<Number> &J_zp_zv = ctxt.get_elem_jacobian(zp_var, zv_var);
	
	DenseSubMatrix<Number> &J_zc_zc = ctxt.get_elem_jacobian(zc_var, zc_var);
	DenseSubMatrix<Number> &J_zc_u = ctxt.get_elem_jacobian(zc_var, u_var);
	DenseSubMatrix<Number> &J_zc_v = ctxt.get_elem_jacobian(zc_var, v_var);
	DenseSubMatrix<Number> &J_zc_c = ctxt.get_elem_jacobian(zc_var, c_var);
	
	DenseSubMatrix<Number> &J_fc_fc = ctxt.get_elem_jacobian(fc_var, fc_var);
	DenseSubMatrix<Number> &J_fc_zc = ctxt.get_elem_jacobian(fc_var, zc_var);

	DenseSubVector<Number> &Ru = ctxt.get_elem_residual( u_var );
	DenseSubVector<Number> &Rv = ctxt.get_elem_residual( v_var );
	DenseSubVector<Number> &Rp = ctxt.get_elem_residual( p_var );
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rzu = ctxt.get_elem_residual( zu_var );
	DenseSubVector<Number> &Rzv = ctxt.get_elem_residual( zv_var );
	DenseSubVector<Number> &Rzp = ctxt.get_elem_residual( zp_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( fc_var );

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
	      c = ctxt.interior_value(c_var, qp),
	      zu = ctxt.interior_value(zu_var, qp),
	      zv = ctxt.interior_value(zv_var, qp),
	      zp = ctxt.interior_value(zp_var, qp),
	      zc = ctxt.interior_value(zc_var, qp),
	      fc = ctxt.interior_value(fc_var, qp);
	    Gradient grad_u = ctxt.interior_gradient(u_var, qp),
	      grad_v = ctxt.interior_gradient(v_var, qp),
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zu = ctxt.interior_gradient(zu_var, qp),
	      grad_zv = ctxt.interior_gradient(zv_var, qp),
	      grad_zc = ctxt.interior_gradient(zc_var, qp),
	      grad_fc = ctxt.interior_gradient(fc_var, qp);

	    // Definitions for convenience.  It is sometimes simpler to do a
	    // dot product if you have the full vector at your disposal.
	    NumberVectorValue U     (u,     v);

	    const Number  u_x = grad_u(0);
	    const Number  v_y = grad_v(1);
	    const Number c_x = grad_c(0);
	    const Number c_y = grad_c(1);
	    const Number zc_x = grad_zc(0);
	    const Number zc_y = grad_zc(1);

	    //things in velocity's family
	    for (unsigned int i=0; i != n_u_dofs; i++){ 
	      Ru(i) += JxW[qp]*(p*dphi[i][qp](0) - (grad_u*dphi[i][qp]));
	      Rv(i) += JxW[qp]*(p*dphi[i][qp](1) - (grad_v*dphi[i][qp]));
	      Rzu(i) += JxW[qp]*(grad_zu*dphi[i][qp] - zp*dphi[i][qp](0) - zc*c_x*phi[i][qp]);
	      Rzv(i) += JxW[qp]*(grad_zv*dphi[i][qp] - zp*dphi[i][qp](1) - zc*c_y*phi[i][qp]);

	      if (request_jacobian && ctxt.elem_solution_derivative){
	        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

	        // Matrix contributions for the uu and vv couplings.
	        for (unsigned int j=0; j != n_u_dofs; j++){ 
	        	Juu(i,j) += JxW[qp]*(-dphi[j][qp]*dphi[i][qp]);
	        	Jvv(i,j) += JxW[qp]*(-dphi[j][qp]*dphi[i][qp]);
	        	
	        	J_zu_zu(i,j) += JxW[qp]*(dphi[j][qp]*dphi[i][qp]);
	        	J_zv_zv(i,j) += JxW[qp]*(dphi[j][qp]*dphi[i][qp]);
	        }

	        // Matrix contributions for the up and vp couplings.
	        for (unsigned int j=0; j != n_p_dofs; j++){
	          Jup(i,j) += JxW[qp]*psi[j][qp]*dphi[i][qp](0);
	          Jvp(i,j) += JxW[qp]*psi[j][qp]*dphi[i][qp](1);
	          
	          J_zu_zp(i,j) += JxW[qp]*(-psi[j][qp]*dphi[i][qp](0));
	          J_zu_zc(i,j) += JxW[qp]*(-psi[j][qp]*c_x*phi[i][qp]);
	          J_zu_c(i,j) += JxW[qp]*(-zc*dpsi[j][qp](0)*phi[i][qp]);
	          J_zv_zp(i,j) += JxW[qp]*(-psi[j][qp]*dphi[i][qp](1));
	          J_zv_zc(i,j) += JxW[qp]*(-psi[j][qp]*c_y*phi[i][qp]);
	          J_zv_c(i,j) += JxW[qp]*(-zc*dpsi[j][qp](1)*phi[i][qp]);
	        }
	      }
	    }
	    
	    //things in pressure's family
	    for (unsigned int i=0; i != n_p_dofs; i++){ 
	      Rp(i) += JxW[qp]*(u_x*psi[i][qp] + v_y*psi[i][qp]);
	      Rc(i) += JxW[qp]*((1.0/parameters[0])*grad_c*dpsi[i][qp] + U*grad_c*psi[i][qp] - fc*psi[i][qp]);
	      Rzp(i) += JxW[qp]*(-zu*dpsi[i][qp](0) - zv*dpsi[i][qp](1));
	      Rzc(i) += JxW[qp]*(grad_zc*dpsi[i][qp] - (U*grad_zc + zc*(u_x + v_y))*psi[i][qp]);
	      if(regtype == 0)
	      	Rfc(i) += JxW[qp]*(beta*fc*psi[i][qp] + zc*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rfc(i) += JxW[qp]*(beta*grad_fc*dpsi[i][qp] + zc*psi[i][qp]);
	      
	      if (request_jacobian && ctxt.elem_solution_derivative){
	        for(unsigned int j=0; j != n_u_dofs; j++){
	          Jpu(i,j) += JxW[qp]*(dphi[j][qp](0)*psi[i][qp]);
	        	Jpv(i,j) += JxW[qp]*(dphi[j][qp](1)*psi[i][qp]);
	        	Jcu(i,j) += JxW[qp]*(phi[j][qp]*c_x*psi[i][qp]);
	        	Jcv(i,j) += JxW[qp]*(phi[j][qp]*c_y*psi[i][qp]);
	        	
	        	J_zp_zu(i,j) += JxW[qp]*(-phi[j][qp]*dpsi[i][qp](0));
	        	J_zp_zv(i,j) += JxW[qp]*(-phi[j][qp]*dpsi[i][qp](1));
	        	J_zc_u(i,j) += JxW[qp]*(-phi[j][qp]*zc_x*psi[i][qp] - zc*dphi[j][qp](0)*psi[i][qp]);
	        	J_zc_v(i,j) += JxW[qp]*(-phi[j][qp]*zc_y*psi[i][qp] - zc*dphi[j][qp](1)*psi[i][qp]);
	       	}
	       	
	       	for(unsigned int j=0; j != n_p_dofs; j++){
	       		Jcc(i,j) += JxW[qp]*((1.0/parameters[0])*dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp]);
	       		
	       		J_c_fc(i,j) += -JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		J_zc_zc(i,j) += JxW[qp]*(dpsi[j][qp]*dpsi[i][qp] - (U*dpsi[j][qp] + psi[j][qp]*(u_x + v_y))*psi[i][qp]);
	       		J_fc_zc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		if(regtype == 0)
	       			J_fc_fc(i,j) += JxW[qp]*(beta*psi[j][qp]*psi[i][qp]);
	       		else if(regtype == 1)
	       			J_fc_fc(i,j) += JxW[qp]*(beta*dpsi[j][qp]*dpsi[i][qp]);
	       	}
	      }
	    }
	  } // end of the quadrature point qp-loop
	  
	  //add contributions from data (from steady_stokes_cd_femsys)
	  /*std::vector<Point> datapts; 
	  std::vector<Real> datavals;
	  if(FILE *fp=fopen("Measurements.dat","r")){
	  	Real x, y, value;
	  	int flag = 1;
	  	while(flag != -1){
	  		flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
	  		if(flag != -1){
					datapts.push_back(Point(x,y));
					datavals.push_back(value);
	  		}
	  	}
	  	fclose(fp);
	  }*/
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
  	  		Rzc(i) += (cstar - cpred)*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_p_dofs; j++)
							J_zc_c(i,j) += -point_phi[j]*point_phi[i] ;
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}


void StokesConvDiffSys::postprocess()
{
  //reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);

}

