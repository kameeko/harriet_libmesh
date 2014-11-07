// This class
#include "grins/practice_cdr_inv.h"

// GRINS
#include "grins/assembly_context.h"
#include "grins/generic_ic_handler.h"
#include "grins/grins_physics_names.h" 
#include "grins/practice_bc_handling.h"
#include "grins/grins_enums.h"

// libMesh
#include "libmesh/quadrature.h"
#include "libmesh/fem_system.h"
#include "libmesh/string_to_enum.h"
#include "libmesh/fe_interface.h"
#include "libmesh/fem_context.h"
#include "libmesh/mesh.h"

namespace GRINS{

	PracticeCDRinv::PracticeCDRinv( const GRINS::PhysicsName& physics_name, const GetPot& input )
		: Physics(physics_name,input), 
			_k(input("Physics/"+physics_name+"/k", 1.0)),
			_R(input("Physics/"+physics_name+"/reaction_coeff", 1.0)),
			_beta(input("Physics/"+physics_name+"/regularization_coeff", 0.01)),
			_fefamily( libMesh::Utility::string_to_enum<GRINSEnums::FEFamily>( 
					input("Physics/"+physics_name+"/fe_family", "LAGRANGE"))){
		
		std::string find_velocity_here = input("Physics/"+physics_name+"/velocity_file","vels0.txt");
		std::string find_data_here = input("Physics/"+physics_name+"/data_file","Measurements.dat");

		if(FILE *fp=fopen(find_velocity_here.c_str(),"r")){
  		libMesh::Real u, v, x, y;
  		libMesh::Real prevx = 1.e10;
  		std::vector<libMesh::Real> tempvecy;
  		std::vector<libMesh::NumberVectorValue> tempvecvel;
  		int flag = 1;
  		while(flag != -1){
  			flag = fscanf(fp, "%lf %lf %lf %lf",&u,&v,&x,&y);
  			if(flag != -1){
  				if(x != prevx){
  					x_pts.push_back(x);
  					prevx = x;
  					if(x_pts.size() > 1){
  						y_pts.push_back(tempvecy);
  						vel_field.push_back(tempvecvel);
						}
  					tempvecy.clear(); 
  					tempvecvel.clear();
  					tempvecy.push_back(y); 
  					tempvecvel.push_back(libMesh::NumberVectorValue(u,v));
  				}
  				else{
  					tempvecy.push_back(y); 
  					tempvecvel.push_back(libMesh::NumberVectorValue(u,v));
  				}
  			}
  		}
  		y_pts.push_back(tempvecy);
  		vel_field.push_back(tempvecvel);
  	}
		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
	  	libMesh::Real x, y, value;
	  	int flag = 1;
	  	while(flag != -1){
	  		flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
	  		if(flag != -1){
					datapts.push_back(libMesh::Point(x,y));
					datavals.push_back(value);
	  		}
	  	}
	  	fclose(fp);
	  }
		this->_bc_handler = new PracticeBCHandling( physics_name, input );
    this->_ic_handler = new GenericICHandler( physics_name, input );
    
		return;
	}
	
	PracticeCDRinv::~PracticeCDRinv(){ return; }
	
	void PracticeCDRinv::init_variables( libMesh::FEMSystem* system ){
		//polynomial order and finite element type for concentration variable
		unsigned int conc_p = 1;
			                                                    
		_c_var = system->add_variable("c", static_cast<GRINSEnums::Order>(conc_p), _fefamily); 
		_zc_var = system->add_variable("zc", static_cast<GRINSEnums::Order>(conc_p), _fefamily); 
		_fc_var = system->add_variable("fc", static_cast<GRINSEnums::Order>(conc_p), _fefamily); 
	}
	
	void PracticeCDRinv::init_context( AssemblyContext& context){
		context.get_element_fe(_c_var)->get_JxW();
    context.get_element_fe(_c_var)->get_phi();
    context.get_element_fe(_c_var)->get_dphi();
    context.get_element_fe(_c_var)->get_xyz();

    context.get_side_fe(_c_var)->get_JxW();
    context.get_side_fe(_c_var)->get_phi();
    context.get_side_fe(_c_var)->get_dphi();
    context.get_side_fe(_c_var)->get_xyz();

    return;
	}
	
	void PracticeCDRinv::element_time_derivative( bool compute_jacobian,
						AssemblyContext& context,
						CachedValues& /*cache*/ ){
	
		// The number of local degrees of freedom in each variable.
    const unsigned int n_c_dofs = context.get_dof_indices(_c_var).size();

    // We get some references to cell-specific data that
    // will be used to assemble the linear system.

    // Element Jacobian * quadrature weights for interior integration.
    const std::vector<libMesh::Real> &JxW =
      context.get_element_fe(_c_var)->get_JxW();

    // The temperature shape function gradients (in global coords.)
    // at interior quadrature points.
    const std::vector<std::vector<libMesh::RealGradient> >& dphi =
      context.get_element_fe(_c_var)->get_dphi();
    const std::vector<std::vector<libMesh::Real> >& phi = context.get_element_fe(_c_var)->get_phi();

    const std::vector<libMesh::Point>& q_points = 
      context.get_element_fe(_c_var)->get_xyz();
    
  	libMesh::DenseSubMatrix<libMesh::Number> &J_c_zc = context.get_elem_jacobian(_c_var, _zc_var);
		libMesh::DenseSubMatrix<libMesh::Number> &J_c_c = context.get_elem_jacobian(_c_var, _c_var);
	
		libMesh::DenseSubMatrix<libMesh::Number> &J_zc_c = context.get_elem_jacobian(_zc_var, _c_var);
		libMesh::DenseSubMatrix<libMesh::Number> &J_zc_fc = context.get_elem_jacobian(_zc_var, _fc_var);
	
		libMesh::DenseSubMatrix<libMesh::Number> &J_fc_zc = context.get_elem_jacobian(_fc_var, _zc_var);
		libMesh::DenseSubMatrix<libMesh::Number> &J_fc_fc = context.get_elem_jacobian(_fc_var, _fc_var);
		
		libMesh::DenseSubVector<libMesh::Number> &Rc = context.get_elem_residual( _c_var );;
		libMesh::DenseSubVector<libMesh::Number> &Rzc = context.get_elem_residual( _zc_var );
		libMesh::DenseSubVector<libMesh::Number> &Rfc = context.get_elem_residual( _fc_var );

    // Now we will build the element Jacobian and residual.
    // Constructing the residual requires the solution and its
    // gradient from the previous timestep.  This must be
    // calculated at each quadrature point by summing the
    // solution degree-of-freedom values by the appropriate
    // weight functions.
    unsigned int n_qpoints = context.get_element_qrule().n_points();

    for (unsigned int qp=0; qp != n_qpoints; qp++){

			libMesh::Number 
	      c = context.interior_value(_c_var, qp),
	      zc = context.interior_value(_zc_var, qp),
	      fc = context.interior_value(_fc_var, qp);
	    libMesh::Gradient 
	      grad_c = context.interior_gradient(_c_var, qp),
	      grad_zc = context.interior_gradient(_zc_var, qp),
	      grad_fc = context.interior_gradient(_fc_var, qp);
			
	  	//location of quadrature point
	  	const libMesh::Real ptx = q_points[qp](0);
	  	const libMesh::Real pty = q_points[qp](1);
			
   		int xind, yind;
   		libMesh::Real xdist = 1.e10; libMesh::Real ydist = 1.e10;
   		for(int ii=0; ii<x_pts.size(); ii++){
   			libMesh::Real tmp = std::abs(ptx - x_pts[ii]);
   			if(xdist > tmp){
   				xdist = tmp;
   				xind = ii;
   			}
   			else
   				break;
   		} 
   		for(int jj=0; jj<y_pts[xind].size(); jj++){
   			libMesh::Real tmp = std::abs(pty - y_pts[xind][jj]);
   			if(ydist > tmp){
   				ydist = tmp;
   				yind = jj;
   			}
   			else
   				break;
   		}
   		libMesh::Real u = vel_field[xind][yind](0);
   		libMesh::Real v = vel_field[xind][yind](1);

	    libMesh::NumberVectorValue U     (u,     v);

	
			// First, an i-loop over the  degrees of freedom.
			for (unsigned int i=0; i != n_c_dofs; i++){
				
				Rc(i) += JxW[qp]*(-_k*grad_zc*dphi[i][qp] + U*grad_zc*phi[i][qp] + 2*_R*zc*c*phi[i][qp]);
	      Rzc(i) += JxW[qp]*(-_k*grad_c*dphi[i][qp] - U*grad_c*phi[i][qp] + _R*c*c*phi[i][qp] + fc*phi[i][qp]);
     		Rfc(i) += JxW[qp]*(_beta*grad_fc*dphi[i][qp] + zc*phi[i][qp]);

				if (compute_jacobian){
					for (unsigned int j=0; j != n_c_dofs; j++){
						J_c_zc(i,j) += JxW[qp]*(-_k*dphi[j][qp]*dphi[i][qp] + U*dphi[j][qp]*phi[i][qp] 
															+ 2*_R*phi[j][qp]*c*phi[i][qp]);
						J_c_c(i,j) += JxW[qp]*(2*_R*zc*phi[j][qp]*phi[i][qp]);

						J_zc_c(i,j) += JxW[qp]*(-_k*dphi[j][qp]*dphi[i][qp] - U*dphi[j][qp]*phi[i][qp] 
																+ 2*_R*c*phi[j][qp]*phi[i][qp]);
						J_zc_fc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
	
	       		J_fc_zc(i,j) += JxW[qp]*(phi[j][qp]*phi[i][qp]);
       			J_fc_fc(i,j) += JxW[qp]*(_beta*dphi[j][qp]*dphi[i][qp]);
					} // end of the inner dof (j) loop
			  } // end - if (compute_jacobian && context.get_elem_solution_derivative())

			} // end of the outer dof (i) loop
    } // end of the quadrature point (qp) loop
    
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	libMesh::Point data_point = datapts[dnum];
	  	if(context.get_elem().contains_point(data_point)){
	  		libMesh::Number cpred = context.point_value(_c_var, data_point);
	  		libMesh::Number cstar = datavals[dnum];
	  		
	  		unsigned int dim = context.get_system().get_mesh().mesh_dimension();
		    libMesh::FEType fe_type = context.get_element_fe(_c_var)->get_fe_type();
		    
		    //go between physical and reference element
		    libMesh::Point c_master = libMesh::FEInterface::inverse_map(dim, fe_type, &context.get_elem(), data_point); 	
		    
        std::vector<libMesh::Real> point_phi(n_c_dofs);
      	for (unsigned int i=0; i != n_c_dofs; i++){
      		//get value of basis function at mapped point in reference (master) element
          point_phi[i] = libMesh::FEInterface::shape(dim, fe_type, &context.get_elem(), i, c_master); 
        }
        
        for (unsigned int i=0; i != n_c_dofs; i++){
  	  		Rc(i) += (cpred - cstar)*point_phi[i];
	  
					if (compute_jacobian){
						for (unsigned int j=0; j != n_c_dofs; j++)
							J_c_c(i,j) += point_phi[j]*point_phi[i] ;
				  }
	  
  			}
	  	}
	  }

    return;
	
	}

} //namespace GRINS
