// General libMesh includes
#include "libmesh/libmesh_common.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"
#include "libmesh/fem_context.h"
#include "libmesh/point.h"
#include "libmesh/quadrature.h"

// Local includes
#include "convdiffstokes_sys.h"
// Bring in everything from the libMesh namespace
using namespace libMesh;

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void StokesConvDiffSys::element_qoi_derivative (DiffContext &context,
                                            const QoISet & /* qois */)
{
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* elem_fe = NULL;
  ctxt.get_element_fe( c_var, elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = elem_fe->get_JxW();

  // The basis functions for the element
  const std::vector<std::vector<Real> > &phi = elem_fe->get_phi();

  // The element quadrature points
  const std::vector<Point > &q_point = elem_fe->get_xyz();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(c_var).size();
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  //QoI depends on concentration, so fill in [0][cvar]-th subderivative...?
  DenseSubVector<Number> &Qc = ctxt.get_qoi_derivatives(0, c_var);
  DenseSubVector<Number> &Qfc = ctxt.get_qoi_derivatives(0, fc_var);

  // Loop over the qps
  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
      const Real x = q_point[qp](0);
      const Real y = q_point[qp](1);

      // If in the sub-domain over which QoI 0 is supported, add contributions
      // to the adjoint rhs
      if(fabs(x - 0.5) <= 0.125 && fabs(y - 0.5) <= 0.125)
        {
        	// Get the solution value at the quadrature point
          Number c = ctxt.interior_value(c_var, qp);
          Number fc = ctxt.interior_value(fc_var, qp);
          
          for (unsigned int i=0; i != n_c_dofs; i++){
            Qc(i) += JxW[qp]*phi[i][qp]*fc;
            Qfc(i) += JxW[qp]*phi[i][qp]*c;
          }
        }

    } // end of the quadrature point qp-loop
}
