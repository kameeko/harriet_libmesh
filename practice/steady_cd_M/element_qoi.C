// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/fe_interface.h"

// Local includes
#include "convdiff_sys.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Define the postprocess function to compute QoI; linearized and evaluated at the same super-state 
//(super-state = states, parameters, adjoints, their auxilary counterparts)

void ConvDiffSys::element_postprocess (DiffContext &context)

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

	    Real u, v;
	    if(vel_option == 0){
	    	u = -(pty-0.5); 
	    	v = ptx-0.5;
	   	}
	   	else if(vel_option == 1){
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
	    NumberVectorValue U     (u,     v);
	    
      dQoI += JxW[qp]*(-(1/params[0])*grad_zc*grad_auxc + U*grad_zc*auxc);
      dQoI += JxW[qp]*(-(1/params[0])*grad_c*grad_auxzc - U*grad_c*auxzc + fc*auxzc);
      if(regtype == 0)
      	dQoI += JxW[qp]*(beta*fc*auxfc + zc*auxfc);
     	else if(regtype == 1)
     		dQoI += JxW[qp]*(beta*grad_fc*grad_auxfc + zc*auxfc);
     		
      dQoI += JxW[qp]*(-(1/params[0])*grad_auxzc*grad_c + U*grad_auxzc*c + auxc*c);
      if(fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125) //is this correct?
   			dQoI += JxW[qp]*c; //?
      dQoI += JxW[qp]*(-(1/params[0])*grad_auxc*grad_zc - U*grad_auxc*zc + auxfc*zc);
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
