// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"
#include "libmesh/fe_interface.h"

// Local includes
#include "convdiff_primary.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

//to compute M_HF(psiLF) and M_LF(psiLF) terms of QoI error estimate

void ConvDiff_PrimarySys::element_postprocess (DiffContext &context)

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
  Number elem_qoi = 0.;

  // Loop over quadrature points
  for (unsigned int qp = 0; qp != n_qpoints; qp++)
    {
      // Get co-ordinate locations of the current quadrature point
      const Real ptx = xyz[qp](0);
      const Real pty = xyz[qp](1);
      
      Number c = ctxt.interior_value(c_var, qp);

			//I(q_LF, u_LF)
      if((qoi_option == 1 && 
						((dim == 2 && (fabs(ptx - 0.5) <= 0.125 && fabs(pty - 0.5) <= 0.125)) || 
						(dim == 1 && ptx >= 0.7 && ptx <= 0.9))) ||
		  		(qoi_option == 2 &&
		  			(dim == 2 && (fabs(ptx - 2.0) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
		  		(qoi_option == 3 &&
		  			(dim == 2 && (fabs(ptx - 0.75) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
	  			(qoi_option == 5) ||
	  			(qoi_option == 6 &&
		    		(dim == 2 && (fabs(ptx - 2.5) <= 0.125 && fabs(pty - 0.5) <= 0.125))) ||
		    	(qoi_option == 7 &&
		    		(dim == 2 && (ptx >= 0.625 && ptx <= 1.5 && fabs(pty - 0.5) <= 0.25 ))) ){	
        elem_qoi += JxW[qp] * c;
			}
    } //end of quadrature loop

  qoi += elem_qoi;

}
