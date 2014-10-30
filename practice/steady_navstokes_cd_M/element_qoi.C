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

// Define the postprocess function to compute QoI; linearized and evaluated at the same super-state 
//(super-state = states, parameters, adjoints, their auxilary counterparts)

void NavStokesConvDiffSys::element_postprocess (DiffContext &context)

{
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* elem_fe = NULL;
  ctxt.get_element_fe( c_var , elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  const std::vector<Point> &xyz = elem_fe->get_xyz();

  const std::vector<Point>& qpoint = elem_fe->get_xyz();

  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  // The function R = int_{omega} T dR
  // omega is a subset of Omega (the whole domain)

  Number dQoI = 0.;

  // Loop over quadrature points

  for (unsigned int qp = 0; qp != n_qpoints; qp++)
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
	      grad_p = ctxt.interior_gradient(p_var, qp),
	      grad_c = ctxt.interior_gradient(c_var, qp),
	      grad_zu = ctxt.interior_gradient(zu_var, qp),
	      grad_zv = ctxt.interior_gradient(zv_var, qp),
	      grad_zp = ctxt.interior_gradient(zp_var, qp),
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
	    const Number p_x = grad_p(0); const Number p_y = grad_p(1);
	    const Number c_x = grad_c(0); const Number c_y = grad_c(1);
	    const Number zc_x = grad_zc(0); const Number zc_y = grad_zc(1);
	    const Number zu_x = grad_zu(0); const Number zu_y = grad_zu(1);
	    const Number zv_x = grad_zv(0); const Number zv_y = grad_zv(1);
	    const Number zp_x = grad_zp(0); const Number zp_y = grad_zp(1);
	   	const Number auxu_x = grad_auxu(0); const Number auxu_y = grad_auxu(1);
	    const Number auxv_x = grad_auxv(0); const Number auxv_y = grad_auxv(1);
	    const Number auxp_x = grad_auxv(0); const Number auxp_y = grad_auxv(1);
	    const Number auxc_x = grad_auxc(0); const Number auxc_y = grad_auxc(1);
	    const Number auxzc_x = grad_auxzc(0); const Number auxzc_y = grad_auxzc(1);
	    const Number auxzu_x = grad_auxzu(0); const Number auxzu_y = grad_auxzu(1);
	    const Number auxzv_x = grad_auxzv(0); const Number auxzv_y = grad_auxzv(1);
	    const Number auxzp_x = grad_auxzv(0); const Number auxzp_y = grad_auxzv(1);
	    
	    dQoI += JxW[qp]*(params[0]*U*grad_u*auxzu - p*auxzu_x + params[1]*grad_u*grad_auxzu);
    	dQoI += JxW[qp]*(params[0]*U*grad_v*auxzv - p*auxzv_y + params[1]*grad_v*grad_auxzv);
    	dQoI += JxW[qp]*(params[1]*grad_zu*grad_auxu - zp*auxu_x - zc*c_x*auxu
														+ params[0]*(-U*grad_zu - v_y*zu)*auxu);
    	dQoI += JxW[qp]*(params[1]*grad_zv*grad_auxv - zp*auxv_y - zc*c_y*auxv
      										+ params[0]*(-U*grad_zv - u_x*zv)*auxv);
    		
     	dQoI += JxW[qp]*(params[0]*(-U*grad_auxzu - v_y*auxzu + auxzv*v_x - auxu*zu_x + zv*auxv_x)*u 
     											+ params[1]*grad_auxzu*grad_u
     											- auxzp*u_x - auxzc*c_x*u - zc*auxc_x*u);
     	dQoI += JxW[qp]*(params[0]*(-U*grad_auxzv - u_x*auxzv + auxzu*u_y - auxv*zv_y + zu*auxu_y)*v
     											+ params[1]*grad_auxzv*grad_v
     											- auxzp*v_y - auxzc*c_y*v - zc*auxc_y*v);
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxu + u_x*auxu)*zu
     											+ params[1]*grad_auxu*grad_zu - auxp*zu_x);
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxv + v_y*auxv)*zv
     											+ params[1]*grad_auxv*grad_zv - auxp*zv_y);
	    
	    
			dQoI += JxW[qp]*(-u_x*auxzp - v_y*auxzp);
      dQoI += JxW[qp]*(-grad_c*grad_auxzc - U*grad_c*auxzc + fc*auxzc);
      dQoI += JxW[qp]*(-zu_x*auxp - zv_y*auxp);
      dQoI += JxW[qp]*(-grad_zc*grad_auxc + (U*grad_zc + zc*(u_x + v_y))*auxc);
      if(regtype == 0)
      	dQoI += JxW[qp]*(beta*fc*auxfc + zc*auxfc);
     	else if(regtype == 1)
     		dQoI += JxW[qp]*(beta*grad_fc*grad_auxfc + zc*auxfc);
     		
     	dQoI += JxW[qp]*(auxzu*p_x + auxzv*p_y);
     	dQoI += JxW[qp]*(-grad_auxzc*grad_c + (U*grad_auxzc + auxzc*(u_x + v_y))*c
   									+ (zc*auxu_x + zc_x*auxu + zc*auxv_y + zc_y*auxv)*c
   									+ auxc*c);
   		if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125) //is this correct?
   			dQoI += JxW[qp];
     	dQoI += JxW[qp]*(auxu*zp_x + auxv*zp_y);
     	dQoI += JxW[qp]*((-auxu*c_x - auxv*c_y + auxfc - U*grad_auxc)*zc
     								- grad_auxc*grad_zc);
     	if(regtype == 0)
     		dQoI += JxW[qp]*((auxzc + beta*auxfc)*fc);
     	else if(regtype == 1)
     		dQoI += JxW[qp]*(auxzc*fc + beta*grad_auxfc*grad_fc);	
     
    }

    for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
			Point data_point = datapts[dnum];
			if(ctxt.get_elem().contains_point(data_point)){
				Number cpred = ctxt.point_value(c_var, data_point);
				Number cstar = datavals[dnum];
				Number auxc = ctxt.point_value(aux_c_var, data_point);
	      
	  		dQoI += (cpred - cstar)*auxc;
			}
		}	


  // Update the computed value of the global functional R, by adding the contribution from this element

  computed_QoI[0] = computed_QoI[0] + dQoI;

}
