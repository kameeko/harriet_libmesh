#include "libmesh/getpot.h"

#include "diff_convdiff_inv.h"

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

#include <cmath>

// Bring in everything from the libMesh namespace
using namespace libMesh;

// System initialization
void Diff_ConvDiff_InvSys::init_data (){
	const unsigned int dim = this->get_mesh().mesh_dimension();

	//polynomial order and finite element type for pressure variable
	unsigned int pressure_p = 1;
	GetPot infile("diff_convdiff_inv.in");
	std::string fe_family = infile("fe_family", std::string("LAGRANGE"));

	// LBB needs better-than-quadratic velocities for better-than-linear
	// pressures, and libMesh needs non-Lagrange elements for
	// better-than-quadratic velocities.
	//libmesh_assert((pressure_p == 1) || (fe_family != "LAGRANGE"));

	FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);
                         
	c_var = this->add_variable("c", static_cast<Order>(pressure_p), fefamily); 
	zc_var = this->add_variable("zc", static_cast<Order>(pressure_p), fefamily); 
	if(dim == 2){
		fc_var = this->add_variable("fc", static_cast<Order>(pressure_p), fefamily); 
		fc1_var = c_var; fc2_var = c_var; fc3_var = c_var; fc4_var = c_var; fc5_var = c_var; 
	}
	else if(dim == 1){
		FEFamily meep = Utility::string_to_enum<FEFamily>(std::string("SCALAR"));
		fc1_var = this->add_variable("fc1", static_cast<Order>(pressure_p), meep);
		fc2_var = this->add_variable("fc2", static_cast<Order>(pressure_p), meep);
		fc3_var = this->add_variable("fc3", static_cast<Order>(pressure_p), meep);
		fc4_var = this->add_variable("fc4", static_cast<Order>(pressure_p), meep);
		fc5_var = this->add_variable("fc5", static_cast<Order>(pressure_p), meep);
		fc_var = c_var;
	}     

	//regularization
	beta = infile("beta", 0.1);
	
	//diffusion coefficient
	k = infile("k", 1.0);
	
	//subdomain ids
	diff_subdomain_id = infile("diff_id", 2);
	convdiff_subdomain_id = infile("convdiff_id", 1);

	//indicate variables that change in time
	this->time_evolving(c_var);
	this->time_evolving(zc_var);
	if(dim == 2)
		this->time_evolving(fc_var);
	else if(dim == 1){
		this->time_evolving(fc1_var); 
		this->time_evolving(fc2_var);
		this->time_evolving(fc3_var);
		this->time_evolving(fc4_var);
		this->time_evolving(fc5_var);
	}
	
	// Useful debugging options
	// Set verify_analytic_jacobians to 1e-6 to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);

	// Set Dirichlet boundary conditions
	//const boundary_id_type all_ids[6] = {0, 1, 2, 3, 4, 5};
	//std::set<boundary_id_type> all_bdys(all_ids, all_ids+(dim*2));
	std::set<boundary_id_type> all_bdys;
	if(dim == 2){ //T-channel
		all_bdys.insert(1); all_bdys.insert(2); all_bdys.insert(3); all_bdys.insert(4);
		all_bdys.insert(5); all_bdys.insert(6); all_bdys.insert(7); all_bdys.insert(8);
	}  
	else if(dim == 1){
		all_bdys.insert(0); all_bdys.insert(1);
	}

	//std::vector<unsigned int> all_of_em(1, c_var);
	std::vector<unsigned int> all_of_em;
	all_of_em.push_back(c_var); all_of_em.push_back(zc_var); all_of_em.push_back(fc_var);
	
	ZeroFunction<Number> zero;
	ConstFunction<Number> one(1);
	
	if(dim == 2){
		//c=0 on boundary, cuz I feel like it...
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, all_of_em, &zero));
	}
	else if(dim == 1){
		std::vector<unsigned int> just_c; just_c.push_back(c_var);
		std::set<boundary_id_type> left_bdy; left_bdy.insert(0);
		std::set<boundary_id_type> right_bdy; right_bdy.insert(1);
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(left_bdy, just_c, &zero));
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(right_bdy, just_c, &one));
		
		std::vector<unsigned int> just_zc; just_zc.push_back(zc_var);
		this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, just_zc, &zero));
	}
	  
	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

// Context initialization
void Diff_ConvDiff_InvSys::init_context(DiffContext &context){
	FEMContext &ctxt = cast_ref<FEMContext&>(context);
  
	//stuff for things of pressure's family
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
bool Diff_ConvDiff_InvSys::element_time_derivative (bool request_jacobian, DiffContext& context){
	const unsigned int dim = this->get_mesh().mesh_dimension();
	Real PI = 3.14159265359;
	
	FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( c_var, c_elem_fe );
  
  int subdomain = ctxt.get_elem().subdomain_id();

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = c_elem_fe->get_JxW();

	const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
	
	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();

	// The subvectors and submatrices we need to fill:
	DenseSubMatrix<Number> &J_c_zc = ctxt.get_elem_jacobian(c_var, zc_var);
	DenseSubMatrix<Number> &J_c_c = ctxt.get_elem_jacobian(c_var, c_var);

	DenseSubMatrix<Number> &J_zc_c = ctxt.get_elem_jacobian(zc_var, c_var);
	DenseSubMatrix<Number> &J_zc_fc = ctxt.get_elem_jacobian(zc_var, fc_var);

	DenseSubMatrix<Number> &J_fc_zc = ctxt.get_elem_jacobian(fc_var, zc_var);
	DenseSubMatrix<Number> &J_fc_fc = ctxt.get_elem_jacobian(fc_var, fc_var);
	
	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( c_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( fc_var );
	
	//for 1D debugging
	DenseSubVector<Number> &Rfc1 = ctxt.get_elem_residual( fc1_var );
	DenseSubVector<Number> &Rfc2 = ctxt.get_elem_residual( fc2_var );
	DenseSubVector<Number> &Rfc3 = ctxt.get_elem_residual( fc3_var );
	DenseSubVector<Number> &Rfc4 = ctxt.get_elem_residual( fc4_var );
	DenseSubVector<Number> &Rfc5 = ctxt.get_elem_residual( fc5_var );
	DenseSubMatrix<Number> &J_zc_fc1 = ctxt.get_elem_jacobian(zc_var, fc1_var);
	DenseSubMatrix<Number> &J_zc_fc2 = ctxt.get_elem_jacobian(zc_var, fc2_var);
	DenseSubMatrix<Number> &J_zc_fc3 = ctxt.get_elem_jacobian(zc_var, fc3_var);
	DenseSubMatrix<Number> &J_zc_fc4 = ctxt.get_elem_jacobian(zc_var, fc4_var);
	DenseSubMatrix<Number> &J_zc_fc5 = ctxt.get_elem_jacobian(zc_var, fc5_var);
	DenseSubMatrix<Number> &J_fc1_fc1 = ctxt.get_elem_jacobian(fc1_var, fc1_var);
	DenseSubMatrix<Number> &J_fc1_fc2 = ctxt.get_elem_jacobian(fc1_var, fc2_var);
	DenseSubMatrix<Number> &J_fc1_fc3 = ctxt.get_elem_jacobian(fc1_var, fc3_var);
	DenseSubMatrix<Number> &J_fc1_fc4 = ctxt.get_elem_jacobian(fc1_var, fc4_var);
	DenseSubMatrix<Number> &J_fc1_fc5 = ctxt.get_elem_jacobian(fc1_var, fc5_var);
	DenseSubMatrix<Number> &J_fc1_zc = ctxt.get_elem_jacobian(fc1_var, zc_var);
	DenseSubMatrix<Number> &J_fc2_fc1 = ctxt.get_elem_jacobian(fc2_var, fc1_var);
	DenseSubMatrix<Number> &J_fc2_fc2 = ctxt.get_elem_jacobian(fc2_var, fc2_var);
	DenseSubMatrix<Number> &J_fc2_fc3 = ctxt.get_elem_jacobian(fc2_var, fc3_var);
	DenseSubMatrix<Number> &J_fc2_fc4 = ctxt.get_elem_jacobian(fc2_var, fc4_var);
	DenseSubMatrix<Number> &J_fc2_fc5 = ctxt.get_elem_jacobian(fc2_var, fc5_var);
	DenseSubMatrix<Number> &J_fc2_zc = ctxt.get_elem_jacobian(fc2_var, zc_var);
	DenseSubMatrix<Number> &J_fc3_fc1 = ctxt.get_elem_jacobian(fc3_var, fc1_var);
	DenseSubMatrix<Number> &J_fc3_fc2 = ctxt.get_elem_jacobian(fc3_var, fc2_var);
	DenseSubMatrix<Number> &J_fc3_fc3 = ctxt.get_elem_jacobian(fc3_var, fc3_var);
	DenseSubMatrix<Number> &J_fc3_fc4 = ctxt.get_elem_jacobian(fc3_var, fc4_var);
	DenseSubMatrix<Number> &J_fc3_fc5 = ctxt.get_elem_jacobian(fc3_var, fc5_var);
	DenseSubMatrix<Number> &J_fc3_zc = ctxt.get_elem_jacobian(fc3_var, zc_var);
	DenseSubMatrix<Number> &J_fc4_fc1 = ctxt.get_elem_jacobian(fc4_var, fc1_var);
	DenseSubMatrix<Number> &J_fc4_fc2 = ctxt.get_elem_jacobian(fc4_var, fc2_var);
	DenseSubMatrix<Number> &J_fc4_fc3 = ctxt.get_elem_jacobian(fc4_var, fc3_var);
	DenseSubMatrix<Number> &J_fc4_fc4 = ctxt.get_elem_jacobian(fc4_var, fc4_var);
	DenseSubMatrix<Number> &J_fc4_fc5 = ctxt.get_elem_jacobian(fc4_var, fc5_var);
	DenseSubMatrix<Number> &J_fc4_zc = ctxt.get_elem_jacobian(fc4_var, zc_var);
	DenseSubMatrix<Number> &J_fc5_fc1 = ctxt.get_elem_jacobian(fc5_var, fc1_var);
	DenseSubMatrix<Number> &J_fc5_fc2 = ctxt.get_elem_jacobian(fc5_var, fc2_var);
	DenseSubMatrix<Number> &J_fc5_fc3 = ctxt.get_elem_jacobian(fc5_var, fc3_var);
	DenseSubMatrix<Number> &J_fc5_fc4 = ctxt.get_elem_jacobian(fc5_var, fc4_var);
	DenseSubMatrix<Number> &J_fc5_fc5 = ctxt.get_elem_jacobian(fc5_var, fc5_var);
	DenseSubMatrix<Number> &J_fc5_zc = ctxt.get_elem_jacobian(fc5_var, zc_var);
	
	// Now we will build the element Jacobian and residual.
	// Constructing the residual requires the solution and its
	// gradient from the previous timestep.  This must be
	// calculated at each quadrature point by summing the
	// solution degree-of-freedom values by the appropriate
	// weight functions.
	unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

	for (unsigned int qp=0; qp != n_qpoints; qp++)
	  {
	    Number 
	      c = ctxt.interior_value(c_var, qp),
	      zc = ctxt.interior_value(zc_var, qp),
	      fc = ctxt.interior_value(fc_var, qp);
	    Gradient 
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zc = ctxt.interior_gradient(zc_var, qp),
	      grad_fc = ctxt.interior_gradient(fc_var, qp);
			
	  	//location of quadrature point
	  	const Real ptx = qpoint[qp](0);
	  	const Real pty = (dim == 2) ? qpoint[qp](1) : 0.0;
			
			//for 1D debug
			Real basis1, basis2, basis3, basis4, basis5;
	    if(dim == 1){
	    	Number f1 = ctxt.interior_value(fc1_var, qp);
	    	Number f2 = ctxt.interior_value(fc2_var, qp);
	    	Number f3 = ctxt.interior_value(fc3_var, qp);
	    	Number f4 = ctxt.interior_value(fc4_var, qp);
	    	Number f5 = ctxt.interior_value(fc5_var, qp);

	    	basis1 = 1.0;
	    	basis2 = sin(2*PI*ptx);
	    	basis3 = cos(2*PI*ptx);
	    	basis4 = sin(4*PI*ptx);
	    	basis5 = cos(4*PI*ptx);
	    	
	    	fc = f_from_coeff(f1, f2, f3, f4, f5, ptx);
	    }
			
			Real u, v;
			if(subdomain == convdiff_subdomain_id && dim == 2){
		 		int xind, yind;
		 		Real xdist = 1.e10; Real ydist = 1.e10;
		 		for(int ii=0; ii<x_pts.size(); ii++){
		 			Real tmp = std::abs(ptx - x_pts[ii]);
		 			if(xdist > tmp){
		 				xdist = tmp;
		 				xind = ii;
		 			}
		 			else
		 				break;
		 		} 
		 		for(int jj=0; jj<y_pts[xind].size(); jj++){
		 			Real tmp = std::abs(pty - y_pts[xind][jj]);
		 			if(ydist > tmp){
		 				ydist = tmp;
		 				yind = jj;
		 			}
		 			else
		 				break;
		 		}
		 		u = vel_field[xind][yind](0);
		 		v = vel_field[xind][yind](1);
   		}
   		else if(subdomain == convdiff_subdomain_id && dim == 1){
   			u = 2.0; v = 0.0;
   		}
   		else if(subdomain == diff_subdomain_id){
   			u = 0.0; v = 0.0;
   		}

	    NumberVectorValue U(u);
	    if(dim == 2)
	    	U(1) = v;
	    	
			Real R = 0.0; //reaction coefficient
	
			// First, an i-loop over the  degrees of freedom.
			for (unsigned int i=0; i != n_c_dofs; i++){
				
				Rc(i) += JxW[qp]*(-k*grad_zc*dphi[i][qp] + U*grad_zc*phi[i][qp] + 2*R*zc*c*phi[i][qp]);
	      Rzc(i) += JxW[qp]*(-k*grad_c*dphi[i][qp] - U*grad_c*phi[i][qp] + R*c*c*phi[i][qp] + fc*phi[i][qp]);
	      if(dim == 2)
     			Rfc(i) += JxW[qp]*(beta*grad_fc*dphi[i][qp] + zc*phi[i][qp]); 
     		else if(dim == 1 && i == 0){ 
		   		Rfc1(i) += JxW[qp]*(beta*basis1*fc + zc*basis1);
		   		Rfc2(i) += JxW[qp]*(beta*basis2*fc + zc*basis2); 
		   		Rfc3(i) += JxW[qp]*(beta*basis3*fc + zc*basis3); 
		   		Rfc4(i) += JxW[qp]*(beta*basis4*fc + zc*basis4); 
		   		Rfc5(i) += JxW[qp]*(beta*basis5*fc + zc*basis5); 
     		}

				if (request_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++){
						J_c_zc(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] + U*dphi[j][qp]*phi[i][qp] 
															+ 2*R*phi[j][qp]*c*phi[i][qp]);
						J_c_c(i,j) += JxW[qp]*(2*R*zc*phi[j][qp]*phi[i][qp]);

						J_zc_c(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] - U*dphi[j][qp]*phi[i][qp] 
																+ 2*R*c*phi[j][qp]*phi[i][qp]);
						if(dim == 2){
							J_zc_fc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
						
			     		J_fc_zc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
		     			J_fc_fc(i,j) += JxW[qp]*(beta*dphi[j][qp]*dphi[i][qp]);
       			}
       			else if(dim == 1){
       				if(j == 0){
		     				J_zc_fc1(i,j) += JxW[qp]*(basis1*phi[i][qp]);
		     				J_zc_fc2(i,j) += JxW[qp]*(basis2*phi[i][qp]);
		     				J_zc_fc3(i,j) += JxW[qp]*(basis3*phi[i][qp]);
		     				J_zc_fc4(i,j) += JxW[qp]*(basis4*phi[i][qp]);
		     				J_zc_fc5(i,j) += JxW[qp]*(basis5*phi[i][qp]);
  						}
  						if(i == 0){
		     				J_fc1_zc(i,j) += JxW[qp]*(phi[j][qp]*basis1);
		     				J_fc2_zc(i,j) += JxW[qp]*(phi[j][qp]*basis2);
		     				J_fc3_zc(i,j) += JxW[qp]*(phi[j][qp]*basis3);
		     				J_fc4_zc(i,j) += JxW[qp]*(phi[j][qp]*basis4);
		     				J_fc5_zc(i,j) += JxW[qp]*(phi[j][qp]*basis5);
		     				
		     				if(j == 0){
				   				J_fc1_fc1(i,j) += JxW[qp]*(beta*basis1*basis1);
				   				J_fc1_fc2(i,j) += JxW[qp]*(beta*basis1*basis2);
				   				J_fc1_fc3(i,j) += JxW[qp]*(beta*basis1*basis3);
				   				J_fc1_fc4(i,j) += JxW[qp]*(beta*basis1*basis4);
				   				J_fc1_fc5(i,j) += JxW[qp]*(beta*basis1*basis5);
				   				J_fc2_fc1(i,j) += JxW[qp]*(beta*basis2*basis1);
				   				J_fc2_fc2(i,j) += JxW[qp]*(beta*basis2*basis2);
				   				J_fc2_fc3(i,j) += JxW[qp]*(beta*basis2*basis3);
				   				J_fc2_fc4(i,j) += JxW[qp]*(beta*basis2*basis4);
				   				J_fc2_fc5(i,j) += JxW[qp]*(beta*basis2*basis5);
				   				J_fc3_fc1(i,j) += JxW[qp]*(beta*basis3*basis1);
				   				J_fc3_fc2(i,j) += JxW[qp]*(beta*basis3*basis2);
				   				J_fc3_fc3(i,j) += JxW[qp]*(beta*basis3*basis3);
				   				J_fc3_fc4(i,j) += JxW[qp]*(beta*basis3*basis4);
				   				J_fc3_fc5(i,j) += JxW[qp]*(beta*basis3*basis5);
				   				J_fc4_fc1(i,j) += JxW[qp]*(beta*basis4*basis1);
				   				J_fc4_fc2(i,j) += JxW[qp]*(beta*basis4*basis2);
				   				J_fc4_fc3(i,j) += JxW[qp]*(beta*basis4*basis3);
				   				J_fc4_fc4(i,j) += JxW[qp]*(beta*basis4*basis4);
				   				J_fc4_fc5(i,j) += JxW[qp]*(beta*basis4*basis5);
				   				J_fc5_fc1(i,j) += JxW[qp]*(beta*basis5*basis1);
				   				J_fc5_fc2(i,j) += JxW[qp]*(beta*basis5*basis2);
				   				J_fc5_fc3(i,j) += JxW[qp]*(beta*basis5*basis3);
				   				J_fc5_fc4(i,j) += JxW[qp]*(beta*basis5*basis4);
				   				J_fc5_fc5(i,j) += JxW[qp]*(beta*basis5*basis5);
		     				}
       				}
       			}
					} // end of the inner dof (j) loop
			  } // end - if (compute_jacobian && context.get_elem_solution_derivative())

			} // end of the outer dof (i) loop
    } // end of the quadrature point (qp) loop
    
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	if(ctxt.get_elem().contains_point(data_point) && (accounted_for[dnum]>=ctxt.get_elem().id()) ){
	  	
	  		//help avoid double-counting if data from edge of elements, but may mess with jacobian check
	  		accounted_for[dnum] = ctxt.get_elem().id(); 
	  		
	  		Number cpred = ctxt.point_value(c_var, data_point);
	  		Number cstar = datavals[dnum];

	  		unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
		    FEType fe_type = ctxt.get_element_fe(c_var)->get_fe_type();
		    
		    //go between physical and reference element
		    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		    
        std::vector<Real> point_phi(n_c_dofs);
      	for (unsigned int i=0; i != n_c_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_c_dofs; i++){
  	  		Rc(i) += (cpred - cstar)*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_c_dofs; j++)
							J_c_c(i,j) += point_phi[j]*point_phi[i] ;
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}

// Postprocessed output
void Diff_ConvDiff_InvSys::postprocess (){

	//reset computed QoIs
  computed_QoI[0] = 0.0;

  FEMSystem::postprocess();

  this->comm().sum(computed_QoI[0]);

}

//calculate forcing function corresponding to basis coefficients; 1D debugging
Real Diff_ConvDiff_InvSys::f_from_coeff(Real fc1, Real fc2, Real fc3, Real fc4, Real fc5, Real x){
	Real PI = 3.14159265359;
	return fc1 + fc2*sin(2*PI*x) + fc3*cos(2*PI*x) + fc4*sin(4*PI*x) + fc5*cos(4*PI*x);
}
