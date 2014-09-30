// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/fe_interface.h"

// Local includes
#include "convdiffnavstokes_sys.h"
// Bring in everything from the libMesh namespace
using namespace libMesh;

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void NavStokesConvDiffSys::element_qoi_derivative (DiffContext &context,
                                            const QoISet & /* qois */)
{
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
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  DenseSubVector<Number> &Qu = ctxt.get_qoi_derivatives(0, u_var);
  DenseSubVector<Number> &Qv = ctxt.get_qoi_derivatives(0, v_var);
  DenseSubVector<Number> &Qp = ctxt.get_qoi_derivatives(0, p_var);
  DenseSubVector<Number> &Qc = ctxt.get_qoi_derivatives(0, c_var);
  DenseSubVector<Number> &Qzu = ctxt.get_qoi_derivatives(0, zu_var);
  DenseSubVector<Number> &Qzv = ctxt.get_qoi_derivatives(0, zv_var);
  DenseSubVector<Number> &Qzp = ctxt.get_qoi_derivatives(0, zp_var);
  DenseSubVector<Number> &Qzc = ctxt.get_qoi_derivatives(0, zc_var);
  DenseSubVector<Number> &Qfc = ctxt.get_qoi_derivatives(0, fc_var);
  DenseSubVector<Number> &Qauxu = ctxt.get_qoi_derivatives(0, aux_u_var);
  DenseSubVector<Number> &Qauxv = ctxt.get_qoi_derivatives(0, aux_v_var);
  DenseSubVector<Number> &Qauxp = ctxt.get_qoi_derivatives(0, aux_p_var);
  DenseSubVector<Number> &Qauxc = ctxt.get_qoi_derivatives(0, aux_c_var);
  DenseSubVector<Number> &Qauxzu = ctxt.get_qoi_derivatives(0, aux_zu_var);
  DenseSubVector<Number> &Qauxzv = ctxt.get_qoi_derivatives(0, aux_zv_var);
  DenseSubVector<Number> &Qauxzp = ctxt.get_qoi_derivatives(0, aux_zp_var);
  DenseSubVector<Number> &Qauxzc = ctxt.get_qoi_derivatives(0, aux_zc_var);
  DenseSubVector<Number> &Qauxfc = ctxt.get_qoi_derivatives(0, aux_fc_var);

  // Loop over the qps
  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
	  	//location of quadrature point
	  	const Real ptx = qpoint[qp](0);
	  	const Real pty = qpoint[qp](1);
	  	
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
	    
	    for (unsigned int i=0; i != n_u_dofs; i++){ 
	      Qu(i) += JxW[qp]*(-params[0]*U*grad_u*phi[i][qp] + p*dphi[i][qp](0) - params[1]*(grad_u*dphi[i][qp]));
	      Qv(i) += JxW[qp]*(-params[0]*U*grad_v*phi[i][qp] + p*dphi[i][qp](1) - params[1]*(grad_v*dphi[i][qp]));
	      Qzu(i) += JxW[qp]*(params[1]*grad_zu*dphi[i][qp] - zp*dphi[i][qp](0) - zc*c_x*phi[i][qp]
	      										+ params[0]*(-U*grad_zu - v_y*zu)*phi[i][qp]);
	      Qzv(i) += JxW[qp]*(params[1]*grad_zv*dphi[i][qp] - zp*dphi[i][qp](1) - zc*c_y*phi[i][qp]
	      										+ params[0]*(-U*grad_zv - u_x*zv)*phi[i][qp]);
	      										
	     	Qauxu(i) += JxW[qp]*(params[0]*(U*grad_auxu + v_y*auxu - auxv*v_x - auxzu*zu_x + zv*auxzv_x)*phi[i][qp] 
	     											- params[1]*grad_auxu*dphi[i][qp]
	     											+ auxp*dphi[i][qp](0) + auxc*c_x*phi[i][qp] + zc*auxzc_x*phi[i][qp]);
	     	Qauxv(i) += JxW[qp]*(params[0]*(U*grad_auxv + u_x*auxv - auxu*u_y - auxzv*zv_y + zu*auxzu_y)*phi[i][qp]
	     											- params[1]*grad_auxv*dphi[i][qp]
	     											+ auxp*dphi[i][qp](1) + auxc*c_y*phi[i][qp] + zc*auxzc_y*phi[i][qp]);
	     	Qauxzu(i) += JxW[qp]*(params[0]*(U*grad_auxzu + u_x*auxzu)*phi[i][qp]
	     											+ params[1]*grad_auxzu*dphi[i][qp] - auxzp*dphi[i][qp](0));
	     	Qauxzv(i) += JxW[qp]*(params[0]*(U*grad_auxzv + v_y*auxzv)*phi[i][qp]
	     											+ params[1]*grad_auxzv*dphi[i][qp] - auxzp*dphi[i][qp](1));
			}
			for (unsigned int i=0; i != n_p_dofs; i++){ 
	      Qp(i) += JxW[qp]*(-u_x*psi[i][qp] + v_y*psi[i][qp]);
	      Qc(i) += JxW[qp]*(grad_c*dpsi[i][qp] + U*grad_c*psi[i][qp] - fc*psi[i][qp]);
	      Qzp(i) += JxW[qp]*(zu*dpsi[i][qp](0) + zv*dpsi[i][qp](1));
	      Qzc(i) += JxW[qp]*(grad_zc*dpsi[i][qp] - (U*grad_zc + zc*(u_x + v_y))*psi[i][qp]);
	      if(regtype == 0)
	      	Qfc(i) += JxW[qp]*(beta*fc*psi[i][qp] + zc*psi[i][qp]);
	     	else if(regtype == 1)
	     		Qfc(i) += JxW[qp]*(beta*grad_fc*dpsi[i][qp] + zc*psi[i][qp]);
	     	Qauxp(i) += JxW[qp]*(-auxu*dpsi[i][qp](0) - auxv*dpsi[i][qp](1));
	     	Qauxc(i) += JxW[qp]*(grad_auxc*dpsi[i][qp] - (U*grad_auxc + auxc*(u_x + v_y))*psi[i][qp]
     									+ (zc*auxzu_x + zc_x*auxzu + zc*auxzv_y + zc_y*auxzv)*psi[i][qp]
     									- auxzc*psi[i][qp]);
     		if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)
     			Qauxc(i) += JxW[qp]*psi[i][qp];
	     	Qauxzp(i) += JxW[qp]*(auxzu*dpsi[i][qp](0) + auxzv*dpsi[i][qp](1));
	     	Qauxzc(i) += JxW[qp]*((-auxzu*c_x - auxzv*c_y + auxfc + U*grad_auxzc)*psi[i][qp]
	     								+ grad_auxzc*dpsi[i][qp]);
	     	if(regtype == 0)
	     		Qauxfc(i) += JxW[qp]*((-auxc + beta*auxfc)*psi[i][qp]);
	     	else if(regtype == 1)
	     		Qauxfc(i) += JxW[qp]*(-auxc*psi[i][qp] + beta*grad_auxfc*dpsi[i][qp]);
   		}
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
			  		Qzc(i) += (cstar - cpred)*point_phi[i];
					}
				}
			}

    } // end of the quadrature point qp-loop
}
