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

//to compute M_HF(psiLF) and M_LF(psiLF) terms of QoI error estimate

void ConvDiff_MprimeSys::element_postprocess (DiffContext &context)

{
	const unsigned int dim = this->get_mesh().mesh_dimension();
	Real PI = 3.14159265359;
	
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* elem_fe = NULL;
  ctxt.get_element_fe( c_var , elem_fe );
  
  int myElemID = ctxt.get_elem().id();

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

	//get location of quadrature points
  const std::vector<Point> &xyz = elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

	//this element's contribution to MHF(psiLF) and MLF(psiLF)
  Number MHF_psiLF_elem = 0.;
  Number MLF_psiLF_elem = 0.;
  
  Number half_sadj_resid_elem = 0.0; //DEBUG

  // Loop over quadrature points
  for (unsigned int qp = 0; qp != n_qpoints; qp++)
    {
      // Get co-ordinate locations of the current quadrature point
      const Real ptx = xyz[qp](0);
      const Real pty = xyz[qp](1);
      
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
      
      int xind, yind;
      Real xdist = 1.e10; Real ydist = 1.e10;
      
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
			
      NumberVectorValue U(u,v);
      
		if(debug_step == 0){ //DEBUG
			//MHF_psiHF = I(q_LF, u_LF) + L'_HF(q_LF, u_LF, z_LF)(p_LF, v_LF, y_LF)
			MHF_psiLF_elem += JxW[qp]*(-k*grad_zc*grad_auxc + U*grad_zc*auxc + 2*R*zc*c*auxc);
      MHF_psiLF_elem += JxW[qp]*(-k*grad_c*grad_auxzc - U*grad_c*auxzc + R*c*c*auxzc + fc*auxzc);
 			MHF_psiLF_elem += JxW[qp]*(beta*grad_fc*grad_auxfc + beta*fc*auxfc + zc*auxfc);

   		if((qoi_option == 1 && 
						((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
						(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
		  		(qoi_option == 2 &&
		  			(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
		  		(qoi_option == 3 &&
		  			(dim == 2 && (fabs(ptx - 0.75) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
	  			(qoi_option == 5) ||
	  			(qoi_option == 6 &&
		    		(dim == 2 && (fabs(ptx - 2.5) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ){	
		  			
        MHF_psiLF_elem += JxW[qp] * c;
			}

			//MLF_psiLF = I(q_LF, u_LF)
      if((qoi_option == 1 && 
						((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
						(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
		  		(qoi_option == 2 &&
		  			(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
		  		(qoi_option == 3 &&
		  			(dim == 2 && (fabs(ptx - 0.75) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
	  			(qoi_option == 5) ||
	  			(qoi_option == 6 &&
		    		(dim == 2 && (fabs(ptx - 2.5) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ){	
		  			
        MLF_psiLF_elem += JxW[qp] * c;
			}
		}
		else if(debug_step == 1){
	     	sadj_c_stash[myElemID].push_back(c);
	      sadj_zc_stash[myElemID].push_back(zc);
	      sadj_fc_stash[myElemID].push_back(fc);
	      sadj_auxc_stash[myElemID].push_back(auxc);
	      sadj_auxzc_stash[myElemID].push_back(auxzc);
	      sadj_auxfc_stash[myElemID].push_back(auxfc);
	      
	      sadj_gradc_stash[myElemID].push_back(grad_c);
	      sadj_gradzc_stash[myElemID].push_back(grad_zc);
	      sadj_gradfc_stash[myElemID].push_back(grad_fc);
	      sadj_gradauxc_stash[myElemID].push_back(grad_auxc);
	      sadj_gradauxzc_stash[myElemID].push_back(grad_auxzc);
	      sadj_gradauxfc_stash[myElemID].push_back(grad_auxfc);
		}
		else if(debug_step == 2){		
			//-0.5*superadj*resid ; half of this needs forward, half of this needs adj...get adj from stash...
			
   		Number 
	      sadj_c = sadj_c_stash[myElemID][qp],
	      sadj_zc = sadj_zc_stash[myElemID][qp],
	      sadj_fc = sadj_fc_stash[myElemID][qp],
	      sadj_auxc = sadj_auxc_stash[myElemID][qp],
	      sadj_auxzc = sadj_auxzc_stash[myElemID][qp],
	      sadj_auxfc = sadj_auxfc_stash[myElemID][qp];
	    Gradient 
	      sadj_grad_c = sadj_gradc_stash[myElemID][qp],
	      sadj_grad_zc = sadj_gradzc_stash[myElemID][qp],
	      sadj_grad_fc = sadj_gradfc_stash[myElemID][qp],
	      sadj_grad_auxc = sadj_gradauxc_stash[myElemID][qp],
	      sadj_grad_auxzc = sadj_gradauxzc_stash[myElemID][qp],
	      sadj_grad_auxfc = sadj_gradauxfc_stash[myElemID][qp];
   		
   		half_sadj_resid_elem += JxW[qp]*(-k*grad_zc*sadj_grad_auxc + U*grad_zc*sadj_auxc + 2*R*zc*c*sadj_auxc);
			half_sadj_resid_elem += JxW[qp]*(-k*grad_c*sadj_grad_auxzc - U*grad_c*sadj_auxzc + R*c*c*sadj_auxzc + fc*sadj_auxzc);
			half_sadj_resid_elem += JxW[qp]*(beta*grad_fc*sadj_grad_auxfc + beta*fc*sadj_auxfc + zc*sadj_auxfc);
      half_sadj_resid_elem += JxW[qp]*(-k*grad_auxzc*sadj_grad_c + U*grad_auxzc*sadj_c 
      						+ 2*R*zc*auxc*sadj_c + 2*R*auxzc*c*sadj_c);
      if((qoi_option == 1 && 
					((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
					(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
				(qoi_option == 2 &&
					(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
	  		(qoi_option == 3 &&
	  			(dim == 2 && (fabs(ptx - 0.75) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
  			(qoi_option == 5) ||
  			(qoi_option == 6 &&
	    		(dim == 2 && (fabs(ptx - 2.5) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ){ 
   			half_sadj_resid_elem += JxW[qp]*sadj_c; 
   		}
      half_sadj_resid_elem += JxW[qp]*(-k*grad_auxc*sadj_grad_zc - U*grad_auxc*sadj_zc 
      						+ auxfc*sadj_zc + 2*R*c*auxc*sadj_zc);
   		half_sadj_resid_elem += JxW[qp]*(auxzc*sadj_fc + beta*grad_auxfc*sadj_grad_fc + beta*auxfc*sadj_fc); 
		}		
		
    } //end of quadrature loop
  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
		Point data_point = datapts[dnum];
		if(ctxt.get_elem().contains_point(data_point)){
			Number cpred = ctxt.point_value(c_var, data_point);
			Number cstar = datavals[dnum];
			
			unsigned int dim = ctxt.get_system().get_mesh().mesh_dimension();
		  FEType fe_type = ctxt.get_element_fe(c_var)->get_fe_type();
		  
		  //go between physical and reference element
		  Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
		  
    	Number auxc_point = ctxt.point_value(aux_c_var, data_point);	      

			if(debug_step == 0)
				MHF_psiLF_elem += (cpred - cstar)*auxc_point;
			else if(debug_step == 1){
				sadj_auxc_point_stash[dnum] = auxc_point;
				sadj_c_point_stash[dnum] = cpred;
			}
			else if(debug_step == 2){
				half_sadj_resid_elem += (cpred - cstar)*sadj_auxc_point_stash[dnum];
				half_sadj_resid_elem += auxc_point*sadj_c_point_stash[dnum];
			}

		}
	}

	if(debug_step == 0){
		MHF_psiLF[myElemID] += MHF_psiLF_elem;
		MLF_psiLF[myElemID] += MLF_psiLF_elem;
	}
	else if(debug_step == 2){
		half_sadj_resid[myElemID] += -0.5*half_sadj_resid_elem; //DEBUG
	}

}
