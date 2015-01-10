// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/fe_interface.h"

// Local includes
#include "convdiff_mprime.h"
// Bring in everything from the libMesh namespace
using namespace libMesh;

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void ConvDiff_MprimeSys::element_qoi_derivative (DiffContext &context,
                                            const QoISet & /* qois */)
{

	const unsigned int dim = this->get_mesh().mesh_dimension();
	Real PI = 3.14159265359;
	
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  //some cell-specific stuff, for u's family and p's family
  FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( c_var, c_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = c_elem_fe->get_JxW();
  
  //at interior quadrature points
  const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
  
  // Physical location of the quadrature points
  const std::vector<Point>& qpoint = c_elem_fe->get_xyz();
  
  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  DenseSubVector<Number> &Qc = ctxt.get_qoi_derivatives(0, c_var);
  DenseSubVector<Number> &Qzc = ctxt.get_qoi_derivatives(0, zc_var);
  DenseSubVector<Number> &Qfc = ctxt.get_qoi_derivatives(0, fc_var);
  DenseSubVector<Number> &Qauxc = ctxt.get_qoi_derivatives(0, aux_c_var);
  DenseSubVector<Number> &Qauxzc = ctxt.get_qoi_derivatives(0, aux_zc_var);
  DenseSubVector<Number> &Qauxfc = ctxt.get_qoi_derivatives(0, aux_fc_var);
  
  //1D DEBUG
  DenseSubVector<Number> &Qfc1 = ctxt.get_qoi_derivatives(0, fc1_var);
  DenseSubVector<Number> &Qfc2 = ctxt.get_qoi_derivatives(0, fc2_var);
  DenseSubVector<Number> &Qfc3 = ctxt.get_qoi_derivatives(0, fc3_var);
  DenseSubVector<Number> &Qfc4 = ctxt.get_qoi_derivatives(0, fc4_var);
  DenseSubVector<Number> &Qfc5 = ctxt.get_qoi_derivatives(0, fc5_var);
  DenseSubVector<Number> &Qauxfc1 = ctxt.get_qoi_derivatives(0, aux_fc1_var);
  DenseSubVector<Number> &Qauxfc2 = ctxt.get_qoi_derivatives(0, aux_fc2_var);
  DenseSubVector<Number> &Qauxfc3 = ctxt.get_qoi_derivatives(0, aux_fc3_var);
  DenseSubVector<Number> &Qauxfc4 = ctxt.get_qoi_derivatives(0, aux_fc4_var);
  DenseSubVector<Number> &Qauxfc5 = ctxt.get_qoi_derivatives(0, aux_fc5_var);

  // Loop over the qps
  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
      
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

      //location of quadrature point
      const Real ptx = qpoint[qp](0);
      const Real pty = qpoint[qp](1);
	  	
	  	//for 1D debug
			Real basis1, basis2, basis3, basis4, basis5;
	    if(dim == 1){
	    	Number f1 = ctxt.interior_value(fc1_var, qp);
	    	Number f2 = ctxt.interior_value(fc2_var, qp);
	    	Number f3 = ctxt.interior_value(fc3_var, qp);
	    	Number f4 = ctxt.interior_value(fc4_var, qp);
	    	Number f5 = ctxt.interior_value(fc5_var, qp);
	    	Number auxf1 = ctxt.interior_value(aux_fc1_var, qp);
	    	Number auxf2 = ctxt.interior_value(aux_fc2_var, qp);
	    	Number auxf3 = ctxt.interior_value(aux_fc3_var, qp);
	    	Number auxf4 = ctxt.interior_value(aux_fc4_var, qp);
	    	Number auxf5 = ctxt.interior_value(aux_fc5_var, qp);

	    	basis1 = 1.0;
	    	basis2 = sin(2*PI*ptx);
	    	basis3 = cos(2*PI*ptx);
	    	basis4 = sin(4*PI*ptx);
	    	basis5 = cos(4*PI*ptx);
	    	
	    	fc = f_from_coeff(f1, f2, f3, f4, f5, ptx);
	    	auxfc = f_from_coeff(auxf1, auxf2, auxf3, auxf4, auxf5, ptx);
	    }
	  	
      Real u, v;
      
      int xind, yind;
      Real xdist = 1.e10; Real ydist = 1.e10;
      
      if(dim == 2){
		    for(int ii=0; ii<x_pts.size(); ii++){
					Real tmp = std::abs(ptx - x_pts[ii]);
					if(xdist > tmp)
						{
						xdist = tmp;
						xind = ii;
					}
					else
						break;
				}
		    
		    for(int jj=0; jj<y_pts[xind].size(); jj++){
					Real tmp = std::abs(pty - y_pts[xind][jj]);
					if(ydist > tmp)
						{
							ydist = tmp;
							yind = jj;
						}
					else
						break;
				}
		    
		    u = vel_field[xind][yind](0);
		    v = vel_field[xind][yind](1);
			}
			else if(dim == 1){
				u = 2.0; v = 0.0;
			}
      NumberVectorValue U(u);
	    if(dim == 2)
	    	U(1) = v;
	    	
      Real R = 0.0; //reaction coefficient
      
      for (unsigned int i=0; i != n_c_dofs; i++){ 
			
				Qauxc(i) += JxW[qp]*(-k*grad_zc*dphi[i][qp] + U*grad_zc*phi[i][qp] + 2*R*zc*c*phi[i][qp]);
			
				Qauxzc(i) += JxW[qp]*(-k*grad_c*dphi[i][qp] - U*grad_c*phi[i][qp] + R*c*c*phi[i][qp] + fc*phi[i][qp]);
				
				if(dim == 2) 	
					Qauxfc(i) += JxW[qp]*(beta*grad_fc*dphi[i][qp] + zc*phi[i][qp]);
				else if(dim == 1 && i == 0){
					Qauxfc1(i) += JxW[qp]*(beta*basis1*fc + zc*basis1);
		   		Qauxfc2(i) += JxW[qp]*(beta*basis2*fc + zc*basis2); 
		   		Qauxfc3(i) += JxW[qp]*(beta*basis3*fc + zc*basis3); 
		   		Qauxfc4(i) += JxW[qp]*(beta*basis4*fc + zc*basis4); 
		   		Qauxfc5(i) += JxW[qp]*(beta*basis5*fc + zc*basis5); 
				}
				 		
				Qc(i) += JxW[qp]*(-k*grad_auxzc*dphi[i][qp] + U*grad_auxzc*phi[i][qp] + 2*R*zc*auxc*phi[i][qp]);
				if((qoi_option == 1 && 
						((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
						(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
		  		(qoi_option == 2 &&
		  			(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125)))){	
	      		
					Qc(i) += JxW[qp]*phi[i][qp]; 
				}
				
				Qzc(i) += JxW[qp]*(-k*grad_auxc*dphi[i][qp] - U*grad_auxc*phi[i][qp] 
					  						+ auxfc*phi[i][qp] + 2*R*c*auxc*phi[i][qp]);
				
				if(dim == 2) 	
					Qfc(i) += JxW[qp]*(auxzc*phi[i][qp] + beta*grad_auxfc*dphi[i][qp]);
				else if(dim == 1 && i == 0){
					Qfc1(i) += JxW[qp]*(beta*basis1*auxfc + auxzc*basis1);
		   		Qfc2(i) += JxW[qp]*(beta*basis2*auxfc + auxzc*basis2); 
		   		Qfc3(i) += JxW[qp]*(beta*basis3*auxfc + auxzc*basis3); 
		   		Qfc4(i) += JxW[qp]*(beta*basis4*auxfc + auxzc*basis4); 
		   		Qfc5(i) += JxW[qp]*(beta*basis5*auxfc + auxzc*basis5); 
				}
			
			} // end loop over n_c_dofs
    } // end of the quadrature point qp-loop
  
  for(unsigned int dnum=0; dnum<datavals.size(); dnum++)
    {
      Point data_point = datapts[dnum];
      if(ctxt.get_elem().contains_point(data_point)){
				Number cpred = ctxt.point_value(c_var, data_point);
				Number cstar = datavals[dnum];
				Number auxc_pointy = ctxt.point_value(aux_c_var, data_point);
			
				unsigned int dim = get_mesh().mesh_dimension();
				FEType fe_type = c_elem_fe->get_fe_type();
					
				//go between physical and reference element
				Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
			
				std::vector<Real> point_phi(n_c_dofs);
				for (unsigned int i=0; i != n_c_dofs; i++)
					{
					  //get value of basis function at mapped point in reference (master) element
				    point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
				  }
			
				for (unsigned int i=0; i != n_c_dofs; i++)
					{
					  Qauxc(i) += (cpred - cstar)*point_phi[i];
					  Qc(i) += auxc_pointy*point_phi[i];
					}
			}
    } // end loop over datavals
} // End element_qoi_derivative
