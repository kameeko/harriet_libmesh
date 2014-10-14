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

	//polynomial order and finite element type for concentration variable
	unsigned int conc_p = 1;
	GetPot infile("convdiff.in");
	std::string fe_family = infile("fe_family", std::string("LAGRANGE"));
	
	Pe = infile("Pe", 1.0);
	R = infile("reaction_coeff", 1.0);
	params.push_back(Pe);

	FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);
	                                                      
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
		not_c_or_f.push_back(aux_c_var); not_c_or_f.push_back(aux_zc_var); ;

	ZeroFunction<Number> zero;
	ConstFunction<Number> one(1);
	  
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
}

// Element residual and jacobian calculations
// Time dependent parts
bool ConvDiffSys::element_time_derivative (bool request_jacobian, DiffContext& context){

	FEMContext &ctxt = cast_ref<FEMContext&>(context);

	//some cell-specific stuff
	FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( c_var, c_elem_fe );

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = c_elem_fe->get_JxW();

	//for concentration, at interior quadrature points
	const std::vector<std::vector<Real> >& psi = c_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dpsi = c_elem_fe->get_dphi();

	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();

	// The subvectors and submatrices we need to fill:
	const unsigned int dim = this->get_mesh().mesh_dimension();

	DenseSubMatrix<Number> &J_c_auxzc = ctxt.get_elem_jacobian(c_var, aux_zc_var);
	DenseSubMatrix<Number> &J_c_auxc = ctxt.get_elem_jacobian(c_var, aux_c_var);
	DenseSubMatrix<Number> &J_c_c = ctxt.get_elem_jacobian(c_var, c_var);
	DenseSubMatrix<Number> &J_c_zc = ctxt.get_elem_jacobian(c_var, zc_var);
	
	DenseSubMatrix<Number> &J_zc_auxc = ctxt.get_elem_jacobian(zc_var, aux_c_var);
	DenseSubMatrix<Number> &J_zc_auxfc = ctxt.get_elem_jacobian(zc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_zc_c = ctxt.get_elem_jacobian(zc_var, c_var);
	
	DenseSubMatrix<Number> &J_fc_auxfc = ctxt.get_elem_jacobian(fc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_fc_auxzc = ctxt.get_elem_jacobian(fc_var, aux_zc_var);
	
	DenseSubMatrix<Number> &J_auxc_zc = ctxt.get_elem_jacobian(aux_c_var, zc_var);
	DenseSubMatrix<Number> &J_auxc_c = ctxt.get_elem_jacobian(aux_c_var, c_var);
	
	DenseSubMatrix<Number> &J_auxzc_c = ctxt.get_elem_jacobian(aux_zc_var, c_var);
	DenseSubMatrix<Number> &J_auxzc_fc = ctxt.get_elem_jacobian(aux_zc_var, fc_var);
	
	DenseSubMatrix<Number> &J_auxfc_zc = ctxt.get_elem_jacobian(aux_fc_var, zc_var);
	DenseSubMatrix<Number> &J_auxfc_fc = ctxt.get_elem_jacobian(aux_fc_var, fc_var);

	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( fc_var );
	DenseSubVector<Number> &Rauxc = ctxt.get_elem_residual( aux_c_var );;
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
	    Number 
	      c = ctxt.interior_value(c_var, qp),
	      zc = ctxt.interior_value(zc_var, qp),
	      fc = ctxt.interior_value(fc_var, qp),
	      auxc = ctxt.interior_value(aux_c_var, qp),
	      auxzc = ctxt.interior_value(aux_zc_var, qp),
	      auxfc = ctxt.interior_value(aux_fc_var, qp);
	    Gradient 
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zc = ctxt.interior_gradient(zc_var, qp),
	      grad_fc = ctxt.interior_gradient(fc_var, qp),
	      grad_auxc = ctxt.interior_gradient(aux_c_var, qp),
	      grad_auxzc = ctxt.interior_gradient(aux_zc_var, qp),
	      grad_auxfc = ctxt.interior_gradient(aux_fc_var, qp);

	    // Definitions for convenience.  It is sometimes simpler to do a
	    // dot product if you have the full vector at your disposal.
	    Real u = -(pty-0.5); Real v = ptx-0.5;
	    NumberVectorValue U     (u,     v);

	    for (unsigned int i=0; i != n_c_dofs; i++){ 
	      Rauxc(i) += JxW[qp]*(-(1/params[0])*grad_zc*dpsi[i][qp] + U*grad_zc*psi[i][qp] + 2*R*zc*c*psi[i][qp]);
	      Rauxzc(i) += JxW[qp]*(-(1/params[0])*grad_c*dpsi[i][qp] - U*grad_c*psi[i][qp] + R*c*c*psi[i][qp] + fc*psi[i][qp]);
	      if(regtype == 0)
	      	Rauxfc(i) += JxW[qp]*(beta*fc*psi[i][qp] + zc*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rauxfc(i) += JxW[qp]*(beta*grad_fc*dpsi[i][qp] + zc*psi[i][qp]);
	     		
	      Rc(i) += JxW[qp]*(-(1/params[0])*grad_auxzc*dpsi[i][qp] + U*grad_auxzc*psi[i][qp] 
	      						+ auxc*psi[i][qp] + 2*R*zc*auxc*psi[i][qp]);
	      if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125) //is this correct?
     			Rc(i) += JxW[qp]*psi[i][qp]; //Rc(i) += JxW[qp]?
	      Rzc(i) += JxW[qp]*(-(1/params[0])*grad_auxc*dpsi[i][qp] - U*grad_auxc*psi[i][qp] 
	      						+ auxfc*psi[i][qp] + 2*R*c*auxc*psi[i][qp]);
	     	if(regtype == 0)
   				Rfc(i) += JxW[qp]*((auxzc + beta*auxfc)*psi[i][qp]);
	     	else if(regtype == 1)
	     		Rfc(i) += JxW[qp]*(auxzc*psi[i][qp] + beta*grad_auxfc*dpsi[i][qp]);

	      if (request_jacobian && ctxt.elem_solution_derivative){
	        libmesh_assert_equal_to (ctxt.elem_solution_derivative, 1.0);

	        for (unsigned int j=0; j != n_c_dofs; j++){

        		J_c_auxzc(i,j) += JxW[qp]*(-(1/params[0])*dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp]);
						J_c_auxc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp] + 2*R*zc*psi[j][qp]*psi[i][qp]);
						J_c_zc(i,j) += JxW[qp]*(2*R*psi[j][qp]*auxc*psi[i][qp]);
     				if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)					
							J_c_c(i,j) += 0; //no dependence on c here if QoI is integral of c over subdomain

						J_zc_auxc(i,j) += JxW[qp]*(-(1/params[0])*dpsi[j][qp]*dpsi[i][qp] - U*dpsi[j][qp]*psi[i][qp]
																+ 2*R*c*psi[j][qp]*psi[i][qp]);
						J_zc_auxfc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
						J_zc_c(i,j) += JxW[qp]*(2*R*psi[j][qp]*auxc*psi[i][qp]);
	
						if(regtype == 0){
			      	J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*psi[j][qp]*psi[i][qp];
	        	}
	        	else if(regtype == 1){
	        		J_fc_auxzc(i,j) += JxW[qp]*(psi[j][qp])*psi[i][qp];
			      	J_fc_auxfc(i,j) += JxW[qp]*beta*dpsi[j][qp]*dpsi[i][qp];
	        	}

						J_auxc_zc(i,j) += JxW[qp]*(-(1/params[0])*dpsi[j][qp]*dpsi[i][qp] + U*dpsi[j][qp]*psi[i][qp] 
																+ 2*R*psi[j][qp]*c*psi[i][qp]);
						J_auxc_c(i,j) += JxW[qp]*(2*R*zc*psi[j][qp]*psi[i][qp]);

						J_auxzc_c(i,j) += JxW[qp]*(-(1/params[0])*dpsi[j][qp]*dpsi[i][qp] - U*dpsi[j][qp]*psi[i][qp] 
																+ 2*R*c*psi[j][qp]*psi[i][qp]);
						J_auxzc_fc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	
	       		J_auxfc_zc(i,j) += JxW[qp]*(psi[j][qp]*psi[i][qp]);
	       		if(regtype == 0)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*psi[j][qp]*psi[i][qp]);
	       		else if(regtype == 1)
	       			J_auxfc_fc(i,j) += JxW[qp]*(beta*dpsi[j][qp]*dpsi[i][qp]);

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
