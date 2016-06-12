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
      const Real ptz = xyz[qp](2);
      
      Number c = ctxt.interior_value(c_var, qp);

			//I(q_LF, u_LF)
      if((qoi_option == 1 && 
    			(dim == 3 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50. && ptz >= 80.))) ||
    		(qoi_option == 1 && 
    			(dim == 2 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50.))) 	){			
        elem_qoi += JxW[qp] * c;
			}
    } //end of quadrature loop

  qoi += elem_qoi;

}
