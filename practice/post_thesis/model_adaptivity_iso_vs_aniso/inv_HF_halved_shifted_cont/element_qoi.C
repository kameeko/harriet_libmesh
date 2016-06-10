// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"

// Local includes
#include "contamTrans_inv.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Define the postprocess function to compute QoI 0

void ContamTransSysInv::element_postprocess (DiffContext &context)

{
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* elem_fe = NULL;
  ctxt.get_element_fe( c_var , elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  const std::vector<Point> &xyz = elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  Number dQoI = 0.;
  
  const unsigned int dim = this->get_mesh().mesh_dimension();

  // Loop over quadrature points
  for (unsigned int qp = 0; qp != n_qpoints; qp++)
    {
      // Get co-ordinate locations of the current quadrature point
      const Real x = xyz[qp](0);
      const Real y = xyz[qp](1);
      const Real z = (dim == 3)?xyz[qp](2):0;

      // If in the sub-domain omega, add the contribution to the integral R
      if((qoi_option == 1 && 
      			(dim == 3 && (fabs(x - 1150.) <= 50. && fabs(y - 825.) <= 50. && z >= 80.))) ||
      		(qoi_option == 1 && 
      			(dim == 2 && (fabs(x - 1150.) <= 50. && fabs(y - 825.) <= 50.))) 	){
      	
        // Get the solution value at the quadrature point
        Number c = ctxt.interior_value(c_var, qp);

        // Update the elemental increment dR for each qp
        dQoI += JxW[qp] * c;
      }
    }

  // Update the computed value of the global functional R, by adding the contribution from this element
  computed_QoI[0] = computed_QoI[0] + dQoI;

}
