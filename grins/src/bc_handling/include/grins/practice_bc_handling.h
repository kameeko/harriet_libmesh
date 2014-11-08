#ifndef GRINS_PRACTICE_BC_HANDLING_H
#define GRINS_PRACTICE_BC_HANDLING_H

//GRINS
#include "grins/bc_handling_base.h"

namespace GRINS
{
  //! Base class for reading and handling boundary conditions for physics classes
  class PracticeBCHandling : public BCHandlingBase
  {
  public:
    
    PracticeBCHandling( const std::string& physics_name, const GetPot& input );
    
    virtual ~PracticeBCHandling();

    virtual void init_bc_data( const libMesh::FEMSystem& system );

    virtual void init_bc_types( const GRINS::BoundaryID bc_id, 
				const std::string& bc_id_string, 
				const int bc_type, 
				const std::string& bc_vars, 
				const std::string& bc_value, 
				const GetPot& input );

    virtual void user_init_dirichlet_bcs( libMesh::FEMSystem* system, 
					  libMesh::DofMap& dof_map,
					  GRINS::BoundaryID bc_id, 
					  GRINS::BCType bc_type ) const;


  private:

    PracticeBCHandling();
    
    VariableIndex _c_var, _zc_var, _fc_var, _aux_c_var, _aux_zc_var, _aux_fc_var;
    bool _has_zc, _has_fc, _has_auxc, _has_auxzc, _has_auxfc;

  };
}
#endif // GRINS_PRACTICE_BC_HANDLING_H
