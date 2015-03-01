#include "libmesh/getpot.h"

#include "convdiff_aux.h"

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
#include "libmesh/equation_systems.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// System initialization
void ConvDiff_AuxSys::init_data (){
	const unsigned int dim = this->get_mesh().mesh_dimension();

	//polynomial order and finite element type
	unsigned int conc_p = 1;
	GetPot infile("convdiff_mprime.in");
	std::string fe_family = infile("fe_family", std::string("LAGRANGE"));

	// LBB needs better-than-quadratic velocities for better-than-linear
	// pressures, and libMesh needs non-Lagrange elements for
	// better-than-quadratic velocities.
	//libmesh_assert((conc_p == 1) || (fe_family != "LAGRANGE"));

	FEFamily fefamily = Utility::string_to_enum<FEFamily>(fe_family);
	aux_c_var = this->add_variable("aux_c", static_cast<Order>(conc_p), fefamily); 
	aux_zc_var = this->add_variable("aux_zc", static_cast<Order>(conc_p), fefamily);
	aux_fc_var = this->add_variable("aux_fc", static_cast<Order>(conc_p), fefamily);   
	
	FEFamily meep = Utility::string_to_enum<FEFamily>(std::string("SCALAR"));
	aux_fpin_var = this->add_variable("aux_fpin", static_cast<Order>(conc_p), meep); 

	//regularization
	beta = infile("beta",0.1);
	
	//diffusion coefficient
	k = infile("k", 1.0);
	
	//reaction coefficient
	R = infile("R", 0.0);
	
	//knobs for how hard to enfore pinning to constant
	screw_mag = infile("aux_mag_screw",1.0e6);
	screw_grad = infile("aux_grad_screw",1.0e0);
	
	//subdomain ids
	scalar_subdomain_id = infile("scalar_id", 2);
	field_subdomain_id = infile("field_id", 1);

	//indicate variables that change in time
	this->time_evolving(aux_c_var);
	this->time_evolving(aux_zc_var);
	this->time_evolving(aux_fc_var);
	this->time_evolving(aux_fpin_var);
	
	// Useful debugging options
	// Set verify_analytic_jacobians to 1e-6 to use
	this->verify_analytic_jacobians = infile("verify_analytic_jacobians", 0.);
	this->print_jacobians = infile("print_jacobians", false);
	this->print_element_jacobians = infile("print_element_jacobians", false);
	this->print_residuals = infile("print_residuals", false);
	this->print_solutions = infile("print_solutions", false);

	// Set Dirichlet boundary conditions
	//const boundary_id_type all_ids[6] = {0, 1, 2, 3, 4, 5};
	//std::set<boundary_id_type> all_bdys(all_ids, all_ids+(dim*2)); std::cout << "\n\nCHANNEL DEBUG\n\n";
	std::set<boundary_id_type> all_bdys;
	
	if(dim == 2){ 
		if(this->get_mesh().get_boundary_info().n_boundary_ids() == 9){ //T-channel
			all_bdys.insert(1); all_bdys.insert(2); all_bdys.insert(3); all_bdys.insert(4);
			all_bdys.insert(5); all_bdys.insert(6); all_bdys.insert(7); all_bdys.insert(8);
		}
		else if(this->get_mesh().get_boundary_info().n_boundary_ids() == 4){ //straight channel
			all_bdys.insert(0); all_bdys.insert(1); all_bdys.insert(2); all_bdys.insert(3);
		}
	}  
	else if(dim == 1){
		all_bdys.insert(0); all_bdys.insert(1);
	}
	
	std::vector<unsigned int> most_of_em;
	most_of_em.push_back(aux_c_var); most_of_em.push_back(aux_zc_var); //homogeneous Neumann on auxfc
	
	ZeroFunction<Number> zero;
	ConstFunction<Number> one(1);

	//c=0 on boundary, cuz I feel like it...
	this->get_dof_map().add_dirichlet_boundary(DirichletBoundary(all_bdys, most_of_em, &zero));
	  
	// Do the parent's initialization after variables and boundary constraints are defined
	FEMSystem::init_data();
}

// Context initialization
void ConvDiff_AuxSys::init_context(DiffContext &context){
	FEMContext &ctxt = cast_ref<FEMContext&>(context);
  
	//stuff for things of pressure's family
	FEBase* c_elem_fe = NULL;

  ctxt.get_element_fe(aux_c_var, c_elem_fe);
  c_elem_fe->get_JxW();
  c_elem_fe->get_phi();
  c_elem_fe->get_dphi();

  FEBase* c_side_fe = NULL;
  ctxt.get_side_fe(aux_c_var, c_side_fe);

  c_side_fe->get_JxW();
  c_side_fe->get_phi();
  c_side_fe->get_dphi();
  
  //add primary solution to the vectors that diff context should localize
  const System & sys = ctxt.get_system();
  NumericVector<Number> &primary_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySys").solution;
 	ctxt.add_localized_vector(primary_solution, sys);
}

// Element residual and jacobian calculations
// Time dependent parts
bool ConvDiff_AuxSys::element_time_derivative (bool request_jacobian, DiffContext& context){
	const unsigned int dim = this->get_mesh().mesh_dimension();
	Real PI = 3.14159265359;

	FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( aux_c_var, c_elem_fe );
  
  int subdomain = ctxt.get_elem().subdomain_id();

	// Element Jacobian * quadrature weights for interior integration
	const std::vector<Real> &JxW = c_elem_fe->get_JxW();

	const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
	const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
	
	// Physical location of the quadrature points
	const std::vector<Point>& qpoint = c_elem_fe->get_xyz();

	// The number of local degrees of freedom in each variable
	const unsigned int n_c_dofs = ctxt.get_dof_indices( aux_c_var ).size();

	// The subvectors and submatrices we need to fill:
	DenseSubMatrix<Number> &J_c_auxzc = ctxt.get_elem_jacobian(aux_c_var, aux_zc_var);
	DenseSubMatrix<Number> &J_c_auxc = ctxt.get_elem_jacobian(aux_c_var, aux_c_var);
	
	DenseSubMatrix<Number> &J_zc_auxc = ctxt.get_elem_jacobian(aux_zc_var, aux_c_var);
	DenseSubMatrix<Number> &J_zc_auxfc = ctxt.get_elem_jacobian(aux_zc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_zc_auxfpin = ctxt.get_elem_jacobian(aux_zc_var, aux_fpin_var);
	
	DenseSubMatrix<Number> &J_fc_auxfc = ctxt.get_elem_jacobian(aux_fc_var, aux_fc_var);
	DenseSubMatrix<Number> &J_fc_auxzc = ctxt.get_elem_jacobian(aux_fc_var, aux_zc_var);
	DenseSubMatrix<Number> &J_fc_auxfpin = ctxt.get_elem_jacobian(aux_fc_var, aux_fpin_var);
	
	DenseSubMatrix<Number> &J_fpin_auxfpin = ctxt.get_elem_jacobian(aux_fpin_var, aux_fpin_var);
	DenseSubMatrix<Number> &J_fpin_auxzc = ctxt.get_elem_jacobian(aux_fpin_var, aux_zc_var);

	DenseSubVector<Number> &Rc = ctxt.get_elem_residual( aux_c_var );
	DenseSubVector<Number> &Rzc = ctxt.get_elem_residual( aux_zc_var );
	DenseSubVector<Number> &Rfc = ctxt.get_elem_residual( aux_fc_var );
	DenseSubVector<Number> &Rfpin = ctxt.get_elem_residual( aux_fpin_var );
	
	
	// Now we will build the element Jacobian and residual.
	// Constructing the residual requires the solution and its
	// gradient from the previous timestep.  This must be
	// calculated at each quadrature point by summing the
	// solution degree-of-freedom values by the appropriate
	// weight functions.
	unsigned int n_qpoints = ctxt.get_element_qrule().n_points();
	
  const System & sys = ctxt.get_system();
  NumericVector<Number> &primary_solution = 
 	 	*const_cast<System &>(sys).get_equation_systems().get_system("ConvDiff_PrimarySys").solution;
  	//*(this->get_equation_systems().get_system("ConvDiff_PrimarySys").solution);
  //std::cout << "\nnorm of primary_solution: " 
  //				<< this->calculate_norm(primary_solution, L2) << " ~~~~~~~~" << std::endl; //DEBUG
  //std::cout << "norm of auxiliary solution: " 
  //				<< this->calculate_norm(*(this->solution), L2) << " ~~~~~~~~" << std::endl; //DEBUG
  std::vector<Number> c_at_qp (n_qpoints, 0);
  std::vector<Number> zc_at_qp (n_qpoints, 0);
  unsigned int c_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySys").variable_number("c");
  unsigned int zc_var = sys.get_equation_systems().get_system("ConvDiff_PrimarySys").variable_number("zc");

  ctxt.interior_values<Number>(c_var, primary_solution, c_at_qp); 
  ctxt.interior_values<Number>(zc_var, primary_solution, zc_at_qp);
  
  //std::cout << "norm of primary_solution: " << this->calculate_norm(primary_solution, L2) << " ~~~~~~~~" << std::endl; //DEBUG
	//std::cout << "norm of auxiliary solution: " 
  //				<< this->calculate_norm(*(this->solution), L2) << " ~~~~~~~~" << std::endl; //DEBUG
	for (unsigned int qp=0; qp != n_qpoints; qp++)
	  {
	    Number 
	      auxc = ctxt.interior_value(aux_c_var, qp),
	      auxzc = ctxt.interior_value(aux_zc_var, qp),
	      auxfc = ctxt.interior_value(aux_fc_var, qp),
	      auxfpin = ctxt.interior_value(aux_fpin_var, qp);
	    Gradient 
	      grad_auxc = ctxt.interior_gradient(aux_c_var, qp),
	      grad_auxzc = ctxt.interior_gradient(aux_zc_var, qp),
	      grad_auxfc = ctxt.interior_gradient(aux_fc_var, qp);
	      
	    Number c = c_at_qp[qp];
	    Number zc = zc_at_qp[qp];
	    //Number c = 1; //DEBUG
	    //Number zc = 1; //DEBUG
	    
	  	//location of quadrature point
	  	const Real ptx = qpoint[qp](0);
	  	const Real pty = qpoint[qp](1);
	  	
	  	//std::cout << ptx << " " << pty << " : " << c << " " << zc << std::endl; //DEBUG
	  	//std::cout << "   " << auxc << " " << auxzc << " " << auxfc << " " << auxfpin << std::endl; //DEBUG
			
			Real u, v;
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

	    NumberVectorValue U(u,v);
	
			// First, an i-loop over the  degrees of freedom.
			for (unsigned int i=0; i != n_c_dofs; i++){
     		
	      Rc(i) += JxW[qp]*(-k*grad_auxzc*dphi[i][qp] + U*grad_auxzc*phi[i][qp] 
	      	+ 2*R*zc*auxc*phi[i][qp] + 2*R*auxzc*c*phi[i][qp]); 
	      if((qoi_option == 1 && 
							((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
							(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
			  		(qoi_option == 2 &&
			  			(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
			  		(qoi_option == 3 &&
			  			(dim == 2 && (fabs(ptx - 0.75) <= 0.125 && fabs(pty - 0.5) <= 0.125)))){			
	      	
     			Rc(i) += JxW[qp]*phi[i][qp]; //Rc(i) += JxW[qp]?
     		}
     		if(subdomain == scalar_subdomain_id){
	      	Rzc(i) += JxW[qp]*(-k*grad_auxc*dphi[i][qp] - U*grad_auxc*phi[i][qp] 
			    						+ auxfpin*phi[i][qp] + 2*R*c*auxc*phi[i][qp]);
			    Rfc(i) += JxW[qp]*(screw_mag*(auxfpin - auxfc)*phi[i][qp] + screw_grad*grad_auxfc*dphi[i][qp]);						
	      }
	      else if(subdomain == field_subdomain_id){
			    Rzc(i) += JxW[qp]*(-k*grad_auxc*dphi[i][qp] - U*grad_auxc*phi[i][qp] 
			    						+ auxfc*phi[i][qp] + 2*R*c*auxc*phi[i][qp]);
		 			Rfc(i) += JxW[qp]*(auxzc*phi[i][qp] + beta*grad_auxfc*dphi[i][qp] + beta*auxfc*phi[i][qp]);
   			}

				if (request_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++){
        		J_c_auxzc(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] + U*dphi[j][qp]*phi[i][qp] + 2*R*phi[j][qp]*c*phi[i][qp]);
						J_c_auxc(i,j) += JxW[qp]*(2*R*zc*phi[j][qp]*phi[i][qp]);
						
						J_zc_auxc(i,j) += JxW[qp]*(-k*dphi[j][qp]*dphi[i][qp] - U*dphi[j][qp]*phi[i][qp]
																+ 2*R*c*phi[j][qp]*phi[i][qp]);
						
						if(subdomain == field_subdomain_id){
							J_zc_auxfc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
						
							J_fc_auxzc(i,j) += JxW[qp]*(phi[j][qp])*phi[i][qp];
					  	J_fc_auxfc(i,j) += JxW[qp]*(beta*dphi[j][qp]*dphi[i][qp] + beta*phi[j][qp]*phi[i][qp]);
						}
						else if(subdomain == scalar_subdomain_id){
							if(j == 0)
								J_zc_auxfpin(i,j) += JxW[qp]*phi[i][qp];
								
							J_fc_auxfc(i,j) += JxW[qp]*(-screw_mag*phi[j][qp]*phi[i][qp] + screw_grad*dphi[j][qp]*dphi[i][qp]);
	     				if(j == 0)
	     					J_fc_auxfpin(i,j) += JxW[qp]*screw_mag*phi[i][qp];
		      	}
		      	
					} // end of the inner dof (j) loop
			  } // end - if (compute_jacobian && context.get_elem_solution_derivative())

			} // end of the outer dof (i) loop
			
			if(subdomain == scalar_subdomain_id){
				Rfpin(0) += JxW[qp]*(beta*auxfpin + auxzc);
				
				if(request_jacobian){
					J_fpin_auxfpin(0,0) += JxW[qp]*beta;
					for (unsigned int j=0; j != n_c_dofs; j++){
						J_fpin_auxzc(0,j) += JxW[qp]*(phi[j][qp]);
					}
				}
			}
    } // end of the quadrature point (qp) loop
    
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	if(ctxt.get_elem().contains_point(data_point) && (accounted_for[dnum]>=ctxt.get_elem().id()) ){
	  	
	  		//help avoid double-counting if data from edge of elements, but may mess with jacobian check
	  		accounted_for[dnum] = ctxt.get_elem().id(); 
	  		
	  		Number auxc_pointy = ctxt.point_value(aux_c_var, data_point);
	  		
	  		unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
		    FEType fe_type = ctxt.get_element_fe(aux_c_var)->get_fe_type();
		    
		    //go between physical and reference element
		    Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		    
        std::vector<Real> point_phi(n_c_dofs);
      	for (unsigned int i=0; i != n_c_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_c_dofs; i++){
  	  		Rc(i) += auxc_pointy*point_phi[i];
	  
					if (request_jacobian){
						for (unsigned int j=0; j != n_c_dofs; j++){
							J_c_auxc(i,j) += point_phi[j]*point_phi[i];
						}
				  }
	  
  			}
	  	}
	  }

	return request_jacobian;
}

