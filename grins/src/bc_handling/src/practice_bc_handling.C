#include "grins/practice_bc_handling.h"

// libMesh
#include "libmesh/const_function.h"
#include "libmesh/dirichlet_boundaries.h"
#include "libmesh/dof_map.h"
#include "libmesh/fem_system.h"

namespace GRINS
{

	PracticeBCHandling::PracticeBCHandling(const std::string& physics_name,
						 const GetPot& input)
    : BCHandlingBase(physics_name){return;}

	PracticeBCHandling::~PracticeBCHandling(){return;}
	
	void PracticeBCHandling::init_bc_data( const libMesh::FEMSystem& system ){
		libmesh_assert( system->has_variable("c");
		_c_var = system->variable_name("c");
		return;
	}
	
	void PracticeBCHandling::init_bc_types(const BoundaryID bc_id, 
					      const std::string& bc_id_string, 
					      const int bc_type, 
					      const std::string& bc_vars, 
					      const std::string& bc_value, 
					      const GetPot& input ){
					      
    this->set_dirichlet_bc_type( bc_id, bc_type );
	  this->set_dirichlet_bc_value( bc_id, 0.0 );
					      
	}
	
	void HeatTransferBCHandling::user_init_dirichlet_bcs( libMesh::FEMSystem* /*system*/,
							libMesh::DofMap& dof_map,
							BoundaryID bc_id,
							BCType bc_type ) const{
	
		std::set<BoundaryID> dbc_ids;
	  dbc_ids.insert(bc_id);
	
	  std::vector<VariableIndex> dbc_vars;
	  dbc_vars.push_back(_c_var);
	
    libMesh::ConstFunction<libMesh::Number> c_func(this->get_dirichlet_bc_value(bc_id));
	
	  libMesh::DirichletBoundary c_dbc( dbc_ids, dbc_vars, &c_func );
	
	  dof_map.add_dirichlet_boundary( c_dbc );	
	
	}

} //namespace GRINS
