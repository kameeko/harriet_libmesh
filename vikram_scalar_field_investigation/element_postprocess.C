#include "scalarfieldsystem.h"
#include "libmesh/fe_base.h"
#include "libmesh/quadrature.h"
#include "libmesh/elem.h"
#include "libmesh/fem_context.h"

#include <iostream>
#include "libmesh/libmesh_common.h"
#include "libmesh/getpot.h"

// Including files for ofstream objects

#include <fstream>
#include <sstream>

#include "libmesh/point_locator_tree.h"
#include "libmesh/point.h"
#include "libmesh/parallel.h"

#include "libmesh/quadrature_gauss.h"
#include "libmesh/quadrature_trap.h"

#define optassert(X) {if (!(X)) libmesh_error();}

// Bring in everything from the libMesh namespace
using namespace libMesh;

void ScalarFieldSystem::element_postprocess(DiffContext &context)
{

  FEMContext &c = libmesh_cast_ref<FEMContext&>(context);

  FEBase* u_elem_fe = NULL;
  c.get_element_fe( 0, u_elem_fe );

  // First we get some references to cell-specific data that
  // will be used to assemble the linear system.

  // Element Jacobian * quadrature weights for interior integration
  const std::vector<Real> &JxW = u_elem_fe->get_JxW();

  const std::vector<Point > &q_point = u_elem_fe->get_xyz();

  // Loop over qp's, compute the function at each qp and add
  // to get the QoI

  unsigned int n_qpoints = c.get_element_qrule().n_points();

  Real dQoI_0 = 0.;
  Real dQoI_1 = 0.;
  Real dQoI_2 = 0.;
  Real dQoI_3 = 0.;
  Real dQoI_4 = 0.;
  Real dQoI_5 = 0.;
  Real dQoI_6 = 0.;
  Real dQoI_7 = 0.;
  Real dQoI_8 = 0.;
  Real dQoI_9 = 0.;
  Real dQoI_10 = 0.;
  Real dQoI_11 = 0.;
  Real dQoI_12 = 0.;
  Real dQoI_13 = 0.;
  Real dQoI_14 = 0.;
  Real dQoI_15 = 0.;
  Real dQoI_16 = 0.;
  Real dQoI_17 = 0.;
  Real dQoI_18 = 0.;
  Real dQoI_19 = 0.;
  Real dQoI_20 = 0.;

  const Real TOL = 1.e-10;

  const Real pie = libMesh::pi;

  // A reference to the system context is built with
  const System & sys = c.get_system();

  // Get a pointer to the adjoint solution vector
  NumericVector<Number> &adjoint_solution = const_cast<System &>(sys).get_adjoint_solution(0);

  // Get the previous adjoint solution values at all the qps

  std::vector<Number> supp_adjoint_c (n_qpoints, 0);
  std::vector<Number> supp_adjoint_zc (n_qpoints, 0);
  std::vector<Number> supp_adjoint_fc (n_qpoints, 0);
  std::vector<Number> supp_adjoint_aux_c (n_qpoints, 0);
  std::vector<Number> supp_adjoint_aux_zc (n_qpoints, 0);
  std::vector<Number> supp_adjoint_aux_fc (n_qpoints, 0);

  c.interior_values<Number>(0, adjoint_solution, supp_adjoint_c);
  c.interior_values<Number>(1, adjoint_solution, supp_adjoint_zc);
  c.interior_values<Number>(2, adjoint_solution, supp_adjoint_fc);
  c.interior_values<Number>(3, adjoint_solution, supp_adjoint_aux_c);
  c.interior_values<Number>(4, adjoint_solution, supp_adjoint_aux_zc);
  c.interior_values<Number>(5, adjoint_solution, supp_adjoint_aux_fc);

  std::vector<Gradient> supp_adjoint_grad_c (n_qpoints, 0);
  std::vector<Gradient> supp_adjoint_grad_zc (n_qpoints, 0);
  std::vector<Gradient> supp_adjoint_grad_fc (n_qpoints, 0);
  std::vector<Gradient> supp_adjoint_grad_aux_c (n_qpoints, 0);
  std::vector<Gradient> supp_adjoint_grad_aux_zc (n_qpoints, 0);
  std::vector<Gradient> supp_adjoint_grad_aux_fc (n_qpoints, 0);

  c.interior_gradient<Number>(0, adjoint_solution, supp_adjoint_grad_c);
  c.interior_gradient<Number>(1, adjoint_solution, supp_adjoint_grad_zc);
  c.interior_gradient<Number>(2, adjoint_solution, supp_adjoint_grad_fc);
  c.interior_gradient<Number>(3, adjoint_solution, supp_adjoint_grad_aux_c);
  c.interior_gradient<Number>(4, adjoint_solution, supp_adjoint_grad_aux_zc);
  c.interior_gradient<Number>(5, adjoint_solution, supp_adjoint_grad_aux_fc);

  for (unsigned int qp=0; qp != n_qpoints; qp++)
    {
      const Real x = q_point[qp](0);

      //Real f_r = 0.0;

      // Get the value of solution at this point
      Number c = c.interior_value(0, qp);
      Number zc = c.interior_value(1, qp);
      Number fc = c.interior_value(2, qp);
      Number aux_c = c.interior_value(3, qp);
      Number aux_zc = c.interior_value(4, qp);
      Number aux_fc = c.interior_value(5, qp);

      Gradient grad_c = c.interior_gradient(0, qp);
      Gradient grad_zc = c.interior_gradient(1, qp);
      Gradient grad_fc = c.interior_gradient(2, qp);
      Gradient grad_aux_c = c.interior_gradient(3, qp);
      Gradient grad_aux_zc = c.interior_gradient(4, qp);
      Gradient grad_aux_fc = c.interior_gradient(5, qp);

      const Number u_x = grad_u(0);
      RealTensor hess_u = c.interior_hessian(0, qp);
      const Number u_xx = hess_u(0,0);

      dQoI_0 += JxW[qp] * beta*( grad_fc* supp_adjoint_aux_fc );

      dQoI_1 += JxW[qp] * ( supp_adjoint_aux_fc*zc );

      dQoI_2 += JxW[qp] * k_d* ( supp_adjoint_grad_aux_c*grad_zc );

      dQoI_3 += JxW[qp] * ( supp_adjoint_aux_c*(grad_zc*U) );
 
      dQoI_4 += JxW[qp] * 2*kr*( c*supp_adjoint_aux_c*zc );

      dQoI_5 += JxW[qp] * ( -k_d*(grad_c*supp_adjoint_grad_aux_zc) + (c*(supp_adjoint_grad_aux_zc*U)) + kr*(c*c*supp_adjoint_aux_zc) ) - ( fc*supp_adjoint_aux_zc );

      dQoI_6 += JxW[qp] * beta*( supp_adjoint_grad_fc*grad_aux_fc );

      dQoI_7 += JxW[qp] * ( supp_adjoint_fc*aux_zc );

      dQoI_8 += JxW[qp] * 2*kr*( supp_adjoint_c*aux_c*zc );

      dQoI_9 += JxW[qp] * ( -k_d*(supp_adjoint_grad_c*grad_aux_zc) + (supp_adjoint_c*(grad_aux_zc*U)) + kr*(supp_adjoint_c*supp_adjoint_c*aux_zc) );

      dQoI_10 += JxW[qp] * ( aux_fc*supp_adjoint_zc ) ;

      dQoI_11 += JxW[qp] * k_d*(grad_aux_c*supp_adjoint_grad_zc);

      dQoI_12 += JxW[qp] * (aux_c*(supp_adjoint_grad_zc*U));

      dQoI_13 += JxW[qp] * 2*kr*(c*aux_c*supp_adjoint_zc);
 


    } // end of the quadrature point qp-loop

  computed_QoI[0] = computed_QoI[0] + dQoI_0;
  computed_QoI[1] = computed_QoI[1] + dQoI_1;
  computed_QoI[2] = computed_QoI[2] + dQoI_2;
  computed_QoI[3] = computed_QoI[3] + dQoI_3;
  computed_QoI[4] = computed_QoI[4] + dQoI_4;
  computed_QoI[5] = computed_QoI[5] + dQoI_5;
  computed_QoI[6] = computed_QoI[6] + dQoI_6;
  computed_QoI[7] = computed_QoI[7] + dQoI_7;
  computed_QoI[8] = computed_QoI[8] + dQoI_8;
  computed_QoI[9] = computed_QoI[9] + dQoI_9;
  computed_QoI[10] = computed_QoI[10] + dQoI_10;
  computed_QoI[11] = computed_QoI[11] + dQoI_11;
  computed_QoI[12] = computed_QoI[12] + dQoI_12;
  computed_QoI[13] = computed_QoI[13] + dQoI_13;
  computed_QoI[14] = computed_QoI[14] + dQoI_14;
  computed_QoI[15] = computed_QoI[15] + dQoI_15;
  computed_QoI[16] = computed_QoI[16] + dQoI_16;
  computed_QoI[17] = computed_QoI[17] + dQoI_17;
  computed_QoI[18] = computed_QoI[18] + dQoI_18;
  computed_QoI[19] = computed_QoI[19] + dQoI_19;
  computed_QoI[20] = computed_QoI[20] + dQoI_20;

}
