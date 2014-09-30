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
	    
      dQoI += JxW[qp]*(-params[0]*U*grad_u*u + p*u_x - params[1]*(grad_u*grad_u));
      dQoI += JxW[qp]*(-params[0]*U*grad_v*v + p*v_y - params[1]*(grad_v*grad_v));
      dQoI += JxW[qp]*(params[1]*grad_zu*grad_zu - zp*zu_x - zc*c_x*zu
      										+ params[0]*(-U*grad_zu - v_y*zu)*zu);
      dQoI += JxW[qp]*(params[1]*grad_zv*grad_zv - zp*zv_y - zc*c_y*zv
      										+ params[0]*(-U*grad_zv - u_x*zv)*zv);
      										
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxu + v_y*auxu - auxv*v_x - auxzu*zu_x + zv*auxzv_x)*auxu 
     											- params[1]*grad_auxu*grad_auxu
     											+ auxp*auxu_x + auxc*c_x*auxu + zc*auxzc_x*auxu);
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxv + u_x*auxv - auxu*u_y - auxzv*zv_y + zu*auxzu_y)*auxv
     											- params[1]*grad_auxv*grad_auxv
     											+ auxp*auxv_y + auxc*c_y*auxv + zc*auxzc_y*auxv);
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxzu + u_x*auxzu)*auxzu
     											+ params[1]*grad_auxzu*grad_auxzu - auxzp*auxzu_x);
     	dQoI += JxW[qp]*(params[0]*(U*grad_auxzv + v_y*auxzv)*auxzv
     											+ params[1]*grad_auxzv*grad_auxzv - auxzp*auxzv_y);

      dQoI += JxW[qp]*(-u_x*p + v_y*p);
      dQoI += JxW[qp]*(grad_c*grad_c + U*grad_c*c - fc*c);
      dQoI += JxW[qp]*(zu*zp_x + zv*zp_y);
      dQoI += JxW[qp]*(grad_zc*grad_zc - (U*grad_zc + zc*(u_x + v_y))*zc);
      if(regtype == 0)
      	dQoI += JxW[qp]*(beta*fc*fc + zc*fc);
     	else if(regtype == 1)
     		dQoI += JxW[qp]*(beta*grad_fc*grad_fc + zc*fc);
     	dQoI += JxW[qp]*(-auxu*auxp_x - auxv*auxp_y);
     	dQoI += JxW[qp]*(grad_auxc*grad_auxc - (U*grad_auxc + auxc*(u_x + v_y))*auxc
   									+ (zc*auxzu_x + zc_x*auxzu + zc*auxzv_y + zc_y*auxzv)*auxc
   									- auxzc*auxc);
   		if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)
   			dQoI += JxW[qp]*auxc;
     	dQoI += JxW[qp]*(auxzu*auxzp_x + auxzv*auxzp_y);
     	dQoI += JxW[qp]*((-auxzu*c_x - auxzv*c_y + auxfc + U*grad_auxzc)*auxzc
     								+ grad_auxzc*grad_auxzc);
     	if(regtype == 0)
     		dQoI += JxW[qp]*((-auxc + beta*auxfc)*auxfc);
     	else if(regtype == 1)
     		dQoI += JxW[qp]*(-auxc*auxfc + beta*grad_auxfc*grad_auxfc);
     
     	for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
				Point data_point = datapts[dnum];
				if(ctxt.get_elem().contains_point(data_point)){
					Number cpred = ctxt.point_value(c_var, data_point);
					Number cstar = datavals[dnum];
		      
		  		dQoI += (cstar - cpred)*cpred;
				}
			}	

    }

  // Update the computed value of the global functional R, by adding the contribution from this element

  computed_QoI[0] = computed_QoI[0] + dQoI;

}
