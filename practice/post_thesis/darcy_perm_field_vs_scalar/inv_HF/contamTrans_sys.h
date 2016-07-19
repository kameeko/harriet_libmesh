// DiffSystem framework files
#include "libmesh/fem_system.h"
#include "libmesh/elem.h"
#include "libmesh/point_locator_tree.h"
#include "libmesh/getpot.h"

using namespace libMesh;

// FEMSystem, TimeSolver and  NewtonSolver will handle most tasks,
// but we must specify element residuals
class ContamTransSys : public FEMSystem
{
public:

  // Constructor
  ContamTransSys(EquationSystems& es, const std::string& name_in, const unsigned int number_in):
    FEMSystem(es, name_in, number_in){
    
    GetPot infile("general.in");
		std::string find_data_here = infile("data_file","Measurements0.dat");
		qoi_option = infile("QoI_option",1);
		
		const unsigned int dim = this->get_mesh().mesh_dimension();

		if(FILE *fp=fopen(find_data_here.c_str(),"r")){
		  if(dim == 3){
				Real x, y, z, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf %lf %lf",&x,&y,&z,&value);
					if(flag != -1){
						datapts.push_back(Point(x,y,z));
						datavals.push_back(value);
					}
				}
				fclose(fp);
	  	}else if(dim == 2){
				Real x, y, value;
				int flag = 1;
				while(flag != -1){
					flag = fscanf(fp,"%lf %lf %lf",&x,&y,&value);
					if(flag != -1){
						datapts.push_back(Point(x,y));
						datavals.push_back(value);
					}
				}
				fclose(fp);
	  	}
	  }
	  //find elements in which data points reside
	  PointLocatorTree point_locator(this->get_mesh());
	  for(unsigned int dnum=0; dnum<datavals.size(); dnum++){
	  	Point data_point = datapts[dnum];
	  	Elem *this_elem = const_cast<Elem *>(point_locator(data_point));
	  	dataelems.push_back(this_elem->id());
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

  // Postprocessed output
  virtual void postprocess ();
  
  //to calculate QoI
  virtual void element_postprocess(DiffContext &context);
  
  //return QoI
  Number &get_QoI_value(std::string type, unsigned int QoI_index){
    return computed_QoI[QoI_index]; //no exact QoI available
  }

  // Indices for each variable;
  unsigned int p_var, z_var, k_var;

  Real dyn_visc; //dynamic viscosity (Pa*s)
  
  Real avg_perm; //m^2
  Real beta; //regularization parameter
  
  //data-related stuff
  std::vector<Point> datapts; 
  std::vector<Real> datavals;
  std::vector<dof_id_type> dataelems;
  
  //to hold computed QoI
  Number computed_QoI[1];
  
  //options for QoI location and nature
  int qoi_option;
  
};
