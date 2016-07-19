// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"

// Local includes
#include "contamTrans_sys.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;

// Define the postprocess function to compute QoI 0

void ContamTransSys::element_postprocess (DiffContext &context)

{
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  FEBase* elem_fe = NULL;
  ctxt.get_element_fe( p_var , elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  const std::vector<Point> &xyz = elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  Number dQoI = 0.;
  
  const unsigned int dim = this->get_mesh().mesh_dimension();
  Real elem_vol = ctxt.get_elem().volume();

  // Loop over quadrature points
  for (unsigned int qp = 0; qp != n_qpoints; qp++)
    {
      // Get the solution value at the quadrature point
      Number p = ctxt.interior_value(p_var, qp);

      // Update the elemental increment dR for each qp
      dQoI += JxW[qp] * p/elem_vol;
    }

  // Update the computed value of the global functional R, by adding the contribution from this element
  computed_QoI[0] = computed_QoI[0] + dQoI;

}
