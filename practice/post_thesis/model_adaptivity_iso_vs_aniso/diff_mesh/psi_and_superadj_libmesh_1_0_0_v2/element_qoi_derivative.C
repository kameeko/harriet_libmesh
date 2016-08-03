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

// We only have one QoI, so we don't bother checking the qois argument
// to see if it was requested from us
void ConvDiff_MprimeSys::element_qoi_derivative (DiffContext &context,
                                            const QoISet & /* qois */)
{

	const unsigned int dim = this->get_mesh().mesh_dimension();
	
  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  //some cell-specific stuff, for u's family and p's family
  FEBase* c_elem_fe = NULL; 
  ctxt.get_element_fe( c_var, c_elem_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = c_elem_fe->get_JxW();
  
  //at interior quadrature points
  const std::vector<std::vector<Real> >& phi = c_elem_fe->get_phi();
  const std::vector<std::vector<RealGradient> >& dphi = c_elem_fe->get_dphi();
  
  // Physical location of the quadrature points
  const std::vector<Point>& qpoint = c_elem_fe->get_xyz();
  
  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices( c_var ).size();
  unsigned int n_qpoints = ctxt.get_element_qrule().n_points();

  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  DenseSubVector<Number> &Qc = ctxt.get_qoi_derivatives(0, c_var);
  DenseSubVector<Number> &Qzc = ctxt.get_qoi_derivatives(0, zc_var);
  DenseSubVector<Number> &Qfc = ctxt.get_qoi_derivatives(0, fc_var);
  DenseSubVector<Number> &Qauxc = ctxt.get_qoi_derivatives(0, aux_c_var);
  DenseSubVector<Number> &Qauxzc = ctxt.get_qoi_derivatives(0, aux_zc_var);
  DenseSubVector<Number> &Qauxfc = ctxt.get_qoi_derivatives(0, aux_fc_var);

  // Loop over the qps
  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
      
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

      //location of quadrature point
      const Real ptx = qpoint[qp](0);
      const Real pty = qpoint[qp](1);
      const Real ptz = qpoint[qp](2);
	  	
      Real R = react_rate; 
			NumberTensorValue k = porosity*dispTens;
			NumberVectorValue U(porosity*vx, 0.0, 0.0);
      
      for (unsigned int i=0; i != n_c_dofs; i++){ 
			
				Qauxc(i) += JxW[qp]*(-k*grad_zc*dphi[i][qp] + U*grad_zc*phi[i][qp] - 2.*R*zc*c*phi[i][qp]);
			
				Qauxzc(i) += JxW[qp]*(-k*grad_c*dphi[i][qp] - U*grad_c*phi[i][qp] - R*c*c*phi[i][qp] + fc*phi[i][qp]);
				
				Qauxfc(i) += JxW[qp]*(beta*grad_fc*dphi[i][qp] + zc*phi[i][qp]);
				 		
				Qc(i) += JxW[qp]*(-k*grad_auxzc*dphi[i][qp] + U*grad_auxzc*phi[i][qp] 
	      	- 2.*R*zc*auxc*phi[i][qp] - 2.*R*auxzc*c*phi[i][qp]);
				if((qoi_option == 0) ||
          (qoi_option == 1 && 
      			(dim == 3 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50. && ptz >= 80.))) ||
      		(qoi_option == 1 && 
      			(dim == 2 && (fabs(ptx - 1150.) <= 50. && fabs(pty - 825.) <= 50.))) 	){			
	      		
					Qc(i) += JxW[qp]*phi[i][qp]; 
				}
				
				Qzc(i) += JxW[qp]*(-k*grad_auxc*dphi[i][qp] - U*grad_auxc*phi[i][qp] 
					  						+ auxfc*phi[i][qp] - 2.*R*c*auxc*phi[i][qp]);
				
				Qfc(i) += JxW[qp]*(auxzc*phi[i][qp] + beta*grad_auxfc*dphi[i][qp]);
			
			} // end loop over n_c_dofs
    } // end of the quadrature point qp-loop
  
  for(unsigned int dnum=0; dnum<datavals.size(); dnum++)
    {
      Point data_point = datapts[dnum];
      if(dataelems[dnum] == ctxt.get_elem().id()){
				Number cpred = ctxt.point_value(c_var, data_point);
				Number cstar = datavals[dnum];
				Number auxc_pointy = ctxt.point_value(aux_c_var, data_point);
			
				unsigned int dim = get_mesh().mesh_dimension();
				FEType fe_type = c_elem_fe->get_fe_type();
					
				//go between physical and reference element
				Point c_master = FEInterface::inverse_map(dim, fe_type, &ctxt.get_elem(), data_point); 	
			  
				std::vector<Real> point_phi(n_c_dofs);
				for (unsigned int i=0; i != n_c_dofs; i++)
					{
					  //get value of basis function at mapped point in reference (master) element
				    point_phi[i] = FEInterface::shape(dim, fe_type, &ctxt.get_elem(), i, c_master); 
				  }
			
				for (unsigned int i=0; i != n_c_dofs; i++)
					{
					  Qauxc(i) += (cpred - cstar)*point_phi[i];
					  Qc(i) += auxc_pointy*point_phi[i];
					}
			}
    } // end loop over datavals
} // End element_qoi_derivative

void ConvDiff_MprimeSys::side_qoi_derivative (DiffContext &context,
                                            const QoISet & /* qois */)
{
  const unsigned int dim = this->get_mesh().mesh_dimension();

  FEMContext &ctxt = cast_ref<FEMContext&>(context);

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.
  FEBase* side_fe = NULL;
  ctxt.get_side_fe(aux_c_var, side_fe );

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = side_fe->get_JxW();

  // Side basis functions
  const std::vector<std::vector<Real> > &phi = side_fe->get_phi();

  // Side Quadrature points
  const std::vector<Point > &qside_point = side_fe->get_xyz();

  //normal vector
  const std::vector<Point> &face_normals = side_fe->get_normals();

  // The number of local degrees of freedom in each variable
  const unsigned int n_c_dofs = ctxt.get_dof_indices(aux_c_var).size();
  
  // Fill the QoI RHS corresponding to this QoI. Since this is the 0th QoI
  // we fill in the [0][i] subderivatives, i corresponding to the variable index.
  DenseSubVector<Number> &Qc = ctxt.get_qoi_derivatives(0, c_var);
  DenseSubVector<Number> &Qauxc = ctxt.get_qoi_derivatives(0, aux_c_var);

  unsigned int n_qpoints = ctxt.get_side_qrule().n_points();

  bool isEast = false;
  if (dim == 3){
    isEast = ctxt.has_side_boundary_id(2);
  }
  else if (dim == 2){
    isEast = ctxt.has_side_boundary_id(1);
  }
  
  //set (in)flux boundary condition on west side
  //homogeneous neumann (Danckwerts) outflow boundary condition on east side
  //no-flux (equivalently, homoegenous neumann) boundary conditions on north, south, top, bottom sides
  //"strong" enforcement of boundary conditions
  for (unsigned int qp=0; qp != n_qpoints; qp++)
  {
    Number z = ctxt.side_value(zc_var, qp);
    Number aux_z = ctxt.side_value(aux_zc_var, qp);

    //velocity vector
    NumberVectorValue U(porosity*vx, 0., 0.); 

    for (unsigned int i=0; i != n_c_dofs; i++)
    {
      if(isEast){
        Qc(i) += JxW[qp]*(-U*face_normals[qp]*aux_z)*phi[i][qp];
        Qauxc(i) += JxW[qp]*(-U*face_normals[qp]*z)*phi[i][qp];
      }
    } //end of outer dof (i) loop
  }

}
