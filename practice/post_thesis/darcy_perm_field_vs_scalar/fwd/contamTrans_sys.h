// DiffSystem framework files
#include "libmesh/fem_system.h"
#include "libmesh/elem.h"
#include "libmesh/point_locator_tree.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class ContamTransSys : public FEMSystem
{
public:

  // Constructor
  ContamTransSys(EquationSystems& es, const std::string& name_in, const unsigned int number_in):
    FEMSystem(es, name_in, number_in){
    
    const unsigned int dim = this->get_mesh().mesh_dimension();
    
    //read in permeability field, assign values to each element (piecewise-constant field)
    std::vector<Point> permpts; 
    std::vector<Real> permvals;
		if(FILE *fp=fopen("true_perm.dat","r")){
		  if(dim == 3){
				Real x, y, z, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf %lf %lf",&x,&y,&z,&value);
					if(flag != -1){
						permpts.push_back(Point(x,y,z));
						permvals.push_back(value);
					}
				}
				fclose(fp);
	  	}else if(dim == 2){
				Real x, y, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
					if(flag != -1){
						permpts.push_back(Point(x,y));
						permvals.push_back(value);
					}
				}
				fclose(fp);
	  	}
	  }
	  //find elements in which data points reside
	  PointLocatorTree point_locator(this->get_mesh());
	  for(unsigned int dnum=0; dnum<permpts.size(); dnum++){
	  	Point data_point = permpts[dnum];
	  	Elem *this_elem = const_cast<Elem *>(point_locator(data_point));
	  	permelems[this_elem->id()] = permvals[dnum];
	  }
    
    }

  // System initialization
  virtual void init_data ();

  // Context initialization
  virtual void init_context(DiffContext &context);

  // Element residual and jacobian calculations
  // Time dependent parts
  virtual bool element_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  //boundary residual and jacobian calculations
  virtual bool side_time_derivative (bool request_jacobian,
                                        DiffContext& context);

  // Postprocessed output
  virtual void postprocess ();

  // Indices for each variable;
  unsigned int p_var;

  Real dyn_visc; //dynamic viscosity (Pa*s)
  std::map<dof_id_type,Real> permelems; //permeability value in each element
  
};