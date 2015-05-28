// Copyright (C) 2014 CCMA@PSU Maximilian Metti, Xiaozhe Hu


//*************************************************
//
//    Still need to improve the Convergence
//    criterion; the current implementation of
//    relative residual will lead to oversolving
//
//**************************************************

#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>
#include "./include/linearized_stokes_with_pnp.h"
#include "./include/linearized_pnp_with_velocity.h"
#include "./include/pressure_mass.h"
#include "./include/poisson_cell_marker.h"
#include "./include/gradient_recovery.h"
extern "C"
{
#include "fasp.h"
#include "fasp_functs.h"
#include "fasp4ns.h"
#include "fasp4ns_functs.h"

    INT fasp_solver_bdcsr_krylov_block_3(block_dCSRmat *A,
                                       dvector *b,
                                       dvector *x,
                                       itsolver_param *itparam,
                                       AMG_param *amgparam,
                                       dCSRmat *A_diag);
    INT fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass (block_dCSRmat *Mat,
                                                                   dvector *b,
                                                                   dvector *x,
                                                                   itsolver_ns_param *itparam,
                                                                   AMG_ns_param *amgnsparam,
                                                                   ILU_param *iluparam,
                                                                   Schwarz_param *schparam,
                                                                   dCSRmat *Mp);
#define FASP_BSR     ON  /** use BSR format in fasp */
#define FASP_NS_MASS ON  /** use solver with pressure mass matrix */
}

double Lx;
double Ly;
double Lz;

using namespace std;
using namespace dolfin;


//////////////////////////////////
//                              //
//      Function Definitions    //
//       of Initial Guesses     //
//                              //
//////////////////////////////////


//  Initial Fluid Velocity Profile
class FluidVelocity : public Expression
{
public:
    FluidVelocity(double out_flow, double in_flow, double bc_dist, int bc_dir): Expression(3),outflow(out_flow),inflow(in_flow),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
        values[bc_direction]  = outflow*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        values[bc_direction] -=  inflow*(x[bc_direction]-bc_distance/2.0)/(bc_distance) + 0.000001*(x[bc_direction]-bc_distance/2.0)/(bc_distance)*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
    }
private:
    double outflow, inflow, bc_distance;
    int bc_direction;
};

//  Initial Charge Carrier Number Density Profile
class LogCharge : public Expression
{
public:
    LogCharge(double ext_bulk, double int_bulk, double bc_dist, int bc_dir): Expression(),ext_contact(ext_bulk),int_contact(int_bulk),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0]  = log(ext_contact)*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        values[0] -= log(int_contact)*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
    }
private:
    double ext_contact, int_contact, bc_distance;
    int bc_direction;
};


//  Voltage
class Voltage : public Expression
{
public:
    Voltage(double ext_volt, double int_volt, double bc_dist, int bc_dir): Expression(),ext_voltage(ext_volt),int_voltage(int_volt),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0]  = ext_voltage*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        values[0] -= int_voltage*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
    }
private:
    double ext_voltage, int_voltage, bc_distance;
    int bc_direction;
};







//  Dirichlet boundary condition
class DirBCval : public Expression
{
public:
    
    DirBCval() : Expression(3) {}
    
    void eval(Array<double>& values, const Array<double>& x) const
    {
        
        values[0] = 0.0;
        values[1] = 0.0;
        values[2] = 0.0;
    }
};

//  Dirichlet boundary condition
// Boundary source for flux boundary condition
class DirNoFlux : public Expression
{
public:

  DirNoFlux(const Mesh& mesh) : Expression(3), mesh(mesh) {}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& ufc_cell) const
  {
    dolfin_assert(ufc_cell.local_facet >= 0);

    Cell cell(mesh, ufc_cell.index);
    Point n = cell.normal(ufc_cell.local_facet);

    const double g = 0.0;
    values[0] = g*n[0];
    values[1] = g*n[1];
    values[2] = g*n[2];
  }

 private:

   const Mesh& mesh;

 };

//////////////////////////////////////
//                                  //
//  Dirichlet Boundary Subdomains   //
//                                  //
//////////////////////////////////////


// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return on_boundary && (x[0] < -Lx+2.*DOLFIN_EPS or x[0] > Lx-2.*DOLFIN_EPS);
    }
};


// Sub domain for Dirichlet boundary condition
class ChannelWalls : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        bool sideWalls = (x[1] > Ly-DOLFIN_EPS or x[1] < -Ly+DOLFIN_EPS);
        bool bottom    = x[2] < -Lz+DOLFIN_EPS;
        bool top       = x[2] >  Lz-DOLFIN_EPS;
        return on_boundary && (sideWalls or bottom or top);
    }
};


// Sub domain for Dirichlet boundary condition
class SideWalls : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return on_boundary && (x[1] > Ly-DOLFIN_EPS and x[1] < -Ly+DOLFIN_EPS);
    }
};

// Sub domain for Dirichlet boundary condition
class TopAndBottom : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return on_boundary && (x[2] > Lz-DOLFIN_EPS and x[2] < -Lz+DOLFIN_EPS);
    }
};



// Sub domain for homogeneous channel wall
class dielectricRegion : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        bool dielectric =  ( fabs(x[0]) < Lx/2.0 );
        
        return ( dielectric );
    }
};




// Sub domain for homogeneous channel wall
class dielectricChannel : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        bool toppatches = ((   (x[0] < -10./3.+DOLFIN_EPS) or (fabs(x[0]+5./6.) < 5./6.+DOLFIN_EPS)
                            or (fabs(x[0]-15./6.) < 5./6.+DOLFIN_EPS))
                           and x[2] > Lz - DOLFIN_EPS  );
        
        bool bottompatches = ((   (x[0] > 10./3.-DOLFIN_EPS) or (fabs(x[0]-5./6.) < 5./6.+DOLFIN_EPS)
                               or (fabs(x[0]+15./6.) < 5./6.+DOLFIN_EPS))
                              and x[2] < -Lz + DOLFIN_EPS  );
        
        return ( on_boundary && (toppatches or bottompatches) );
        //                &&  (x[1] < -Ly+DOLFIN_EPS or x[1] > Ly-DOLFIN_EPS
        //                   or   x[2] < -Lz+DOLFIN_EPS or x[2] > Lz-DOLFIN_EPS) );
    }
};


// Sub domain for homogeneous channel wall
class channelGate : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return ( on_boundary && (x[2] < -Lz + DOLFIN_EPS or x[2] > Lz - DOLFIN_EPS) );
        //                && (x[1] < -Ly+DOLFIN_EPS or x[1] > Ly-DOLFIN_EPS
        //                  or  x[2] < -Lz+DOLFIN_EPS or x[2] > Lz-DOLFIN_EPS) );
    }
};




//////////////////////////
//                      //
//      Main Program    //
//                      //
//////////////////////////

int main()
{
    
    ////////////////////////////////////
    //                                //
    //    Setup the environment       //
    //    initialize the problem      //
    //                                //
    ////////////////////////////////////
    
    // Set linear algebra backend
    parameters["linear_algebra_backend"] = "uBLAS";
    parameters["allow_extrapolation"] = true;
    
    printf(" \n---------------------------------------------\n"); fflush(stdout);
    printf(" This code adaptively solves Stokes' \n"); fflush(stdout);
    printf(" equation for a constant viscosity \n"); fflush(stdout);
    printf(" using a Discontinuous Galerkin discretization  \n"); fflush(stdout);
    printf("------------------------------------------------\n\n"); fflush(stdout);
    
    
    
    
    printf(" \n-------------------------------------\n"); fflush(stdout);
    printf(" Initializing the problem \n"); fflush(stdout);
    printf("-------------------------------------\n\n"); fflush(stdout);
    
    
    //***********************
    //  Read in parameters
    //***********************
    
    printf("Read in parameters for the solver and describing the PDE\n");
    fflush(stdout);
    char buffer[500];            // max number of char for each line
    int  val;
    ifstream expin;
    char paramRegime[128];       // output directory
    char meshIn[128];            // mesh input file
    char surfIn[128];            // mesh surfaces file
    char subdIn[128];            // mesh subdomains file
    char point_charge_file[128]; // point charge position file

    // BoxMesh Parameters
    double Dx, Dy, Dz, T;
    int    Nx, Ny, Nz, Nt;

    // Newton Solver Parameters
    double tol;
    uint   maxit;
    double mu;
    double adaptTol;
    uint   max_bGS_it;
    double bGS_tol;
    
    // Boundary conditions
    int    bc_direction;
    double bc_distance;
    double in_flow, out_flow;
    double ext_voltage, int_voltage;
    double ext_cat_bulk, int_cat_bulk;
    double ext_an_bulk,  int_an_bulk;

    // PDE coefficients
    double temperature;
    double viscosity;
    double density;
    double penalty;
    double perm_p, perm_n;
    double cat_diff_p, cat_diff_n, cat_valency;
    double an_diff_p, an_diff_n, an_valency;
    double fix_p, fix_n;
    
    char filenm[] = "./params/electrokinetic-params.dat";
    
    // if input file is not specified, use the default values
    /*if (filenm=="") {
        printf("### ERROR: No file specified for input params \n");
        exit(0);
    }*/
    
    FILE *fp = fopen(filenm,"r");
    if (fp==NULL) {
        printf("### ERROR: Could not open file %s...\n", filenm);
        fasp_chkerr(ERROR_OPEN_FILE, "fasp_param_input");
        
    }
    
    bool state = true;
    while ( state ) {
        int     ibuff;
        double  dbuff;
        char    sbuff[500];
        char   *fgetsPtr;
        
        val = fscanf(fp,"%s",buffer);
        if (val==EOF) break;
        if (val!=1){ state = false; break; }
        if (buffer[0]=='[' || buffer[0]=='%' || buffer[0]=='|') {
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
            continue;
        }
        
        // match keyword and scan for value
        if (strcmp(buffer,"outdir")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { state = false; break; }
            strncpy(paramRegime,sbuff,128);
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"mesh_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { state = false; break; }
            strncpy(meshIn,sbuff,128);
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"surf_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { state = false; break; }
            strncpy(surfIn,sbuff,128);
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"subd_file")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%s",sbuff);
            if (val!=1) { state = false; break; }
            strncpy(subdIn,sbuff,128);
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"x_length")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            Dx = dbuff; Lx = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"y_length")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            Dy = dbuff; Ly = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"z_length")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            Dz = dbuff; Lz = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"t_length")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            T = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"x_grid")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            Nx = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"y_grid")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            Ny = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"z_grid")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            Nz = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"t_grid")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            Nt = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        
        else if (strcmp(buffer,"nonlin_tol")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            tol = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"nonlin_maxit")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            maxit = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"block_GS_tol")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            bGS_tol = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"block_GS_maxit")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            max_bGS_it = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"damp_factor")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            mu = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"adapt_tol")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            adaptTol = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"bc_direction")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%d",&ibuff);
            if (val!=1) { state = false; break; }
            bc_direction = ibuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"bc_distance")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            bc_distance = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"fluid_outflow")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            out_flow = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"fluid_inflow")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            in_flow = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"density")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            density = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"ext_voltage")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            ext_voltage = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"int_voltage")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            int_voltage = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"ext_cat_bulk")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            ext_cat_bulk = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"int_cat_bulk")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            int_cat_bulk = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"ext_an_bulk")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            ext_an_bulk = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"int_an_bulk")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            int_an_bulk = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"temperature")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            temperature = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"viscosity")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            viscosity = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"penalty")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            penalty = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"perm_p")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            perm_p = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"perm_n")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            perm_n = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"cat_diff_p")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            cat_diff_p = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"cat_diff_n")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            cat_diff_n = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"cat_valency")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            cat_valency = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"an_diff_p")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            an_diff_p = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"an_diff_n")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            an_diff_n = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"an_valency")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            an_valency = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }

        else if (strcmp(buffer,"fix_p")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            fix_p = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else if (strcmp(buffer,"fix_n")==0) {
            val = fscanf(fp,"%s",buffer);
            if (val!=1 || strcmp(buffer,"=")!=0) {
                state = false; break;
            }
            val = fscanf(fp,"%lf",&dbuff);
            if (val!=1) { state = false; break; }
            fix_n = dbuff;
            fgetsPtr = fgets(buffer,500,fp); // skip rest of line
        }
        
        else {
            state = false;
            printf(" Bad read-in \n\n"); fflush(stdout);
        }
        
        
    }
    
    fclose(fp);
    
    printf("\nSolver parameters \n"); fflush(stdout);
    printf("\t adaptivity tol           = %e \n",adaptTol); fflush(stdout);
    printf("\t nonlinear solver tol     = %e \n",tol); fflush(stdout);
    printf("\t nonlinear solver max it  = %d \n",maxit); fflush(stdout);
    printf("\t nonlinear solver damping = %e \n\n",mu); fflush(stdout);
    printf("\t DG penalty term          = %e \n\n",penalty); fflush(stdout);
    
    
    
    //*****************************
    //  Open files to write data
    //*****************************
    
    File meshfile("./output/electrokinetics/mesh.pvd");
    File  Velocityfile("./output/electrokinetics/VelFinal.pvd");
    File  Pressurefile("./output/electrokinetics/PressFinal.pvd");
    File  Catfile("./output/electrokinetics/CatFinal.pvd");
    File   Anfile("./output/electrokinetics/AnFinal.pvd");
    File  phifile("./output/electrokinetics/phiFinal.pvd");
    
    
    //***********
    //  Domain
    //***********
    
    printf("Initialize mesh: "); fflush(stdout);
    Mesh initMesh;
    MeshFunction<std::size_t> surfaces;
    MeshFunction<std::size_t> subdoms;

    if ( strcmp(meshIn,"box")==0 ) {
      printf(" Domain set to [ %6.3f, %6.3f, %6.3f ] \n\n", Dx, Dy, Dz);
      printf(" Initial mesh is %d x %d x %d \n", Nx,Ny,Nz); fflush(stdout);
      BoxMesh bMesh(-Lx,-Ly,-Lz, Lx, Ly, Lz, Nx, Ny, Nz);
      initMesh = bMesh;
    } else {
      printf(" Reading in the mesh from %s \n", meshIn);
        Mesh rMesh(meshIn);
        initMesh = rMesh;
        printf(" Reading in the mesh surfaces from %s \n", surfIn);
        MeshFunction<std::size_t> surfReadIn(initMesh, surfIn);
        printf(" Reading in the mesh subdomains from %s \n", subdIn);
        MeshFunction<std::size_t>  subdReadIn(initMesh, subdIn);
          surfaces = surfReadIn;
          subdoms  = subdReadIn;
          //meshfile << surfaces;
          //meshfile << subdoms;
    }
    
    
    
    
    //************************
    //  Dimensional Analysis
    //************************
    
    // Reference values (universal constants)
    double k_B    = 1.38064880e-23 ;    // Boltzmann Constant (m^2 kg / s^2 K)
    double eps_0  = 8.85418782e-12 ;    // Vacuum Permittivity (s^4 A^2 / m^3 kg)
    double e_chrg = 1.60217657e-19 ;    // Elementary Positive Charge (A s = C)
    double n_avo  = 6.02214129e+23 ;    // Avogadro's number 1 / mol


    // Units of measurement             **** MUST MATCH INPUT FILE ****
    double L_ref  = 1.0e-09;            //  length scale (m)
    double p_ref  = 1.0e+00;            //  ionic reference density (mol / m^3)
    double v_ref  = 1.334e+0;           //  velocity reference (m / s)


    // Derive scaling for coefficients
    double m_ref     = L_ref * n_avo * p_ref * k_B * temperature / v_ref;      // Viscosity (Pa s = kg / m s)
    double D_ref     = L_ref * v_ref;                                          // reference diffusivity (m^2 / s)
    double d_ref     = n_avo * p_ref * k_B * temperature / (v_ref*v_ref);      // reference fluid density (kg / m^3)
    double press_ref = n_avo * p_ref * k_B * temperature;                      // reference pressure (kg / m^2 s)
    double debye, epsPVal, epsNVal;

    debye = sqrt( (k_B*temperature*eps_0)/(e_chrg*e_chrg*n_avo*p_ref) ) / L_ref; // Debye Length
    
    printf(" The spatial scale is %e meters   \n", L_ref); fflush(stdout);
    printf(" The temporal scale is %e seconds \n", L_ref/v_ref); fflush(stdout);
    printf("   The (dim'less) debye length is %e \n\n", debye); fflush(stdout);
    
    
    // PDE coefficients
    viscosity  = viscosity / m_ref;   // dim'less viscosity
    density    = density / d_ref;     // dim'less fluid density
    epsPVal    = perm_p * debye;      // dim'less permittivity in P-doped material
    epsNVal    = perm_n * debye;      // dim'less permittivity in N-doped material
    cat_diff_p = cat_diff_p / D_ref;  // dim'less cation diffusivity in P-doped material
    cat_diff_n = cat_diff_n / D_ref;  // dim'less cation diffusivity in N-doped material
    an_diff_p  = an_diff_p  / D_ref;  // dim'less anion diffusivity in P-doped material
    an_diff_n  = an_diff_n  / D_ref;  // dim'less anion diffusivity in N-doped material
    fix_p      = fix_p / p_ref;       // dim'less fixed charge in P-doped material
    fix_n      = fix_n / p_ref;       // dim'less fixed charge in N-doped material
    
    printf(" Dimensionless parameters are given by: \n"); fflush(stdout);
    printf("    Fluidic Viscosity:          \t %15.10f \n", viscosity); fflush(stdout);
    printf("    Fluidic Density:            \t %15.10f \n", density); fflush(stdout);
    printf("    P-doped Permittivity:       \t %15.10f \n", epsPVal); fflush(stdout);
    printf("    N-doped Permittivity:       \t %15.10f \n", epsNVal); fflush(stdout);
    printf("    P-doped Cation Diffusivity: \t %15.10f \n", cat_diff_p); fflush(stdout);
    printf("    N-doped Cation Diffusivity: \t %15.10f \n", cat_diff_n); fflush(stdout);
    printf("    P-doped Anion Diffusivity:  \t %15.10f \n", an_diff_p); fflush(stdout);
    printf("    N-doped Anion Diffusivity:  \t %15.10f \n", an_diff_n); fflush(stdout);
    printf("    P-doping:                   \t %15.10f \n", fix_p); fflush(stdout);
    printf("    N-doping:                   \t %15.10f \n", fix_n); fflush(stdout);
    fflush(stdout);
    
    
    // Boundary conditions
    in_flow  = in_flow  / v_ref;                               // dim'less inflow rate
    out_flow = out_flow / v_ref;                               // dim'less outflow rate
    int_voltage = int_voltage / (k_B*temperature/e_chrg);      // dim'less internal contact voltage
    ext_voltage = ext_voltage / (k_B*temperature/e_chrg);      // dim'less external contact voltage
    int_cat_bulk = int_cat_bulk / p_ref;                       // dim'less internal contact cation
    ext_cat_bulk = ext_cat_bulk / p_ref;                       // dim'less external contact cation
    int_an_bulk  = int_an_bulk  / p_ref;                       // dim'less internal contact anion
    ext_an_bulk  = ext_an_bulk  / p_ref;                       // dim'less external contact anion
    
    printf("\n The boundary conditions are: \n"); fflush(stdout);
    printf("    Fluidic Rate of Inflow:        %15.10f m/s \n", -in_flow); fflush(stdout);
    printf("    Fluidic Rate of Outflow:       %15.10f m/s \n", out_flow); fflush(stdout);
    printf("    Voltage at External Contact:   %15.10f V \n", ext_voltage); fflush(stdout);
    printf("    Voltage at Internal Contact:   %15.10f V \n", int_voltage); fflush(stdout);
    printf("    Cation  at External Contact:   %15.10f M \n", ext_cat_bulk); fflush(stdout);
    printf("    Cation  at Internal Contact:   %15.10f M \n", int_cat_bulk); fflush(stdout);
    printf("    Anion   at External Contact:   %15.10f M \n", ext_an_bulk);  fflush(stdout);
    printf("    Anion   at Internal Contact:   %15.10f M \n", int_an_bulk);  fflush(stdout);
    
    
    
    
    
    //************************
    //  Analytic Expressions
    //************************
    printf("\n Initializing analytic expressions\n"); fflush(stdout);

    // Velocity
    printf("   Interpolating in/out flow rates for fluid \n"); fflush(stdout);
      FluidVelocity Velocity(out_flow,in_flow,bc_distance,bc_direction);
    
    // Log-ion boundary interpolant
    printf("   Interpolating contact values for charge carriers \n"); fflush(stdout);
      LogCharge Cation(ext_cat_bulk,int_cat_bulk,bc_distance,bc_direction);
      LogCharge Anion(  ext_an_bulk, int_an_bulk,bc_distance,bc_direction);

    // Electric potential boundary interpolant
    printf("   Interpolating voltage drop\n"); fflush(stdout);
      Voltage Volt(ext_voltage,int_voltage,bc_distance,bc_direction);
    
    
    
    //***************
    //  Time March
    //***************
    
    
    
    
    
    
    
    //**************
    //  Adaptivity
    //**************

    
    // Copy mesh and initialize CG space
    Mesh mesh0(initMesh);
    linearized_stokes_with_pnp::FunctionSpace VQ0(mesh0);
    linearized_pnp_with_velocity::FunctionSpace W0(mesh0);
    uint totalRefines = 0;
    
    // Define initial guesses and residual
    printf("\n Define initial guesses\n"); fflush(stdout);
    Function initGuessNS(VQ0);
    Constant zero(0.0);
    Constant zero_vec(0.0,0.0,0.0);
    Function initVeloc(initGuessNS[0]); initVeloc.interpolate(Velocity);
    Function initPress(initGuessNS[1]); initPress.interpolate(zero);

    Function initGuess(W0);
    Function initCat(initGuess[0]); initCat.interpolate(Cation);
    Function initAn(initGuess[1]);  initAn.interpolate(Anion);
    Function initPHI(initGuess[2]); initPHI.interpolate(Volt);
    
    // Initialize adaptivity marker
    Function adaptFn0(W0);
    Function adaptFN(adaptFn0[2]);
    adaptFN.interpolate(initPHI);
    double b0_stokes_fasp = -1.0;
    double b0_fasp = -1.0;
    bool converged = false;
    
    
    // Adapt mesh to current iterate
    for ( uint adaptInd=0; adaptInd<15; adaptInd++ ) {
    
    bool tooCoarse  = true;
    bool unrefined  = true;
    uint numRefines = 0;
    Mesh mesh(mesh0);
    
        
    // Coarsen until all elements are sufficiently refined        
    if ( strcmp(meshIn,"box")==0 ) {
        printf("\n Refine mesh until recovered gradient is sufficiently accurate\n");
        fflush(stdout);
    } else {
        printf("\n Skipping refinement for given mesh\n"); fflush(stdout);
        tooCoarse=false;
    }
        
    while (tooCoarse) {
        
        // Gradient recovery
        printf(" Recovering gradient of electric potential... \n"); fflush(stdout);
          Mesh meshAdapt(mesh);
          gradient_recovery::FunctionSpace GR(meshAdapt);
          gradient_recovery::BilinearForm aGR(GR,GR);
          gradient_recovery::LinearForm LGR(GR);
          LGR.u = adaptFN;
          Function Dsoln(GR);
          //solve(aGR==LGR,Dsoln);
        
        
        //**************************************************
        //  Solve linear problem: interface with FASP
        //**************************************************
    
        Matrix adaptA; assemble(adaptA,aGR);
        Vector adaptb; assemble(adaptb,LGR);
        Function adaptSolu(GR);
        
        printf(" -------------------------------------\n"); fflush(stdout);
        printf(" Start interface to FASP  \n"); fflush(stdout);
        printf(" -------------------------------------\n\n"); fflush(stdout);
        
        printf(" Step 1: convert sparse matrix format and lump\n"); fflush(stdout);
        // Convert adaptA to CSR
        dCSRmat adaptA_fasp;
        
        unsigned int adaptnz = boost::tuples::get<3>(adaptA.data());
        int adaptrow = adaptA.size(0);
        int adaptcol = adaptA.size(1);
        int* adaptap = (int*)fasp_mem_calloc(adaptrow+1, sizeof(int));
        const size_t* ap_tmp = boost::tuples::get<0>(adaptA.data());
        for (int i=0; i<adaptrow+1; i++) {
            adaptap[i] = (int)ap_tmp[i];
        }
        int* adaptai = (int*)fasp_mem_calloc(adaptnz, sizeof(int));
        const size_t* ai_tmp = boost::tuples::get<1>(adaptA.data());
        for (int i=0; i<adaptnz; i++) {
            adaptai[i] = (int)ai_tmp[i];
        }
        double* adaptax = (double*)boost::tuples::get<2>(adaptA.data());
        
        // Lump matrix
        for ( uint rowInd=0; rowInd<adaptrow; rowInd++ ) {
            double  rowSum = 0.;
            int diagColInd = -1;
            for ( uint colInd=adaptap[rowInd]; colInd < adaptap[rowInd+1]; colInd++ ) {
                rowSum += adaptax[colInd];
                adaptax[colInd] = 0.0;
                if ( adaptai[colInd] == rowInd ) diagColInd = colInd;
            }
            adaptax[diagColInd] = rowSum;
            //printf(" adapt_A[%d,%d] = %e \n", rowInd, adaptai[diagColInd], adaptax[diagColInd]);
        }
        
        adaptA_fasp.row = adaptrow;
        adaptA_fasp.col = adaptcol;
        adaptA_fasp.nnz = adaptnz;
        adaptA_fasp.IA  = adaptap;
        adaptA_fasp.JA  = adaptai;
        adaptA_fasp.val = adaptax;
        
        // initialize RHS
        dvector adaptb_fasp;
        dvector b_fasp;
        adaptb_fasp.row = adaptb.size();
        adaptb_fasp.val = (double*)adaptb.data();
        
        // initialize solution
        dvector adaptsoluvec;
        fasp_dvec_alloc(adaptb_fasp.row, &adaptsoluvec);
        fasp_dvec_set(adaptb_fasp.row, &adaptsoluvec, 0.0);
        
        
        // Need solver for generic 3x3 Mass matrix
        //#if FASP_BSR
        // convert CSR to BSR
        dBSRmat adaptA_fasp_bsr = fasp_format_dcsr_dbsr(&adaptA_fasp, 3);
        
        // free CSR matrix
        //fasp_dcsr_free(&A_fasp);
        
        
        printf(" Step 2: initialize solver parameters\n"); fflush(stdout);
        // initialize solver parameters
        input_param     inpar;  // parameters from input files
        itsolver_param  itpar;  // parameters for itsolver
        AMG_param       amgpar; // parameters for AMG
        ILU_param       ilupar; // parameters for ILU
        //#endif
        
        // read in parameters from a input file
        //#if FASP_BSR
        char inputfile[] = "./params/bsr.dat";
        fasp_param_input(inputfile, &inpar);
        fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);
        
        
        printf(" Step 3: solve the linear system\n"); fflush(stdout);
        // solve
        int status=FASP_SUCCESS;
        //fasp_param_amg_print(&amgpar);
        
        //#if FASP_BSR
        status = fasp_solver_dbsr_krylov_diag(&adaptA_fasp_bsr, &adaptb_fasp, &adaptsoluvec, &itpar);
        //status = fasp_solver_dbsr_krylov_amg(&adaptA_fasp_bsr, &adaptb_fasp, &adaptsoluvec, &itpar, &amgpar);
        //#endif
        
        
        
        if (status<0) {
            printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", status); fflush(stdout);
        }
        else {
            printf("\nSolver finished successfully!\n\n"); fflush(stdout);
        }
        
        
        
        printf(" Step 4: convert solution back\n"); fflush(stdout);
        // Convert solution vector to FE solution
        double * adaptSolVal = adaptSolu.vector()->data();
        //#if FASP_BSR
        for(std::size_t i=0; i<adaptsoluvec.row; ++i) {
            adaptSolVal[i] = adaptsoluvec.val[i];
        }
        //#endif
        Dsoln = adaptSolu;
        
        
        
        
        printf(" Step 5: free memory\n"); fflush(stdout);
        // Free memory
        //#if FASP_BSR
        //fasp_dbsr_free(&adaptA_fasp_bsr);
        //#endif
        //fasp_dvec_free(&b_fasp);
        fasp_dvec_free(&adaptsoluvec);
        free(adaptap); free(adaptai);
        
        printf(" -------------------------------------\n"); fflush(stdout);
        printf(" End of interface to FASP \n"); fflush(stdout);
        printf(" -------------------------------------\n\n"); fflush(stdout);
     
        

        
        // Estimate error
        printf(" Estimating error... "); fflush(stdout);
          poisson_cell_marker::FunctionSpace DG(meshAdapt);
          poisson_cell_marker::LinearForm errForm(DG);
          errForm.Du = Dsoln;
          errForm.u  = adaptFN;
        
        // Mark elements for refinement
        printf(" Marking elements for refinement\n "); fflush(stdout);
          Vector errVec;
          assemble(errVec,errForm);
          uint refineCount=0;
          MeshFunction<bool> cellMark(meshAdapt,3,false);
          for ( uint errVecInd=0; errVecInd<errVec.size(); errVecInd++) {
            if ( errVec[errVecInd] > adaptTol ) {
                refineCount++;
                cellMark.values()[errVecInd] = true;
            }
          }
        
        // Refine marked elemetns
        if ( refineCount>0 ) {
            printf(" Refine mesh\n "); fflush(stdout);
            mesh = refine(meshAdapt,cellMark);
            numRefines++;
            unrefined  = false;
        }
        
        // No elements marked for refinement
        else {
            printf("No elements marked for refinement... adaptivity complete\n");
            fflush(stdout);
            tooCoarse = false;
        }
        
    }
        
    // Solve complete
    if ( unrefined && converged ) {
        printf(" Converged to a solution \n\n"); fflush(stdout);
        printf(" Review output files \n\n"); fflush(stdout);
        return 0;
    }
    
    // Adapt Subdomains and Surfaces
    std::shared_ptr<const Mesh> new_mesh( new const Mesh(mesh) );
    MeshFunction<std::size_t> adapted_surfaces;
    MeshFunction<std::size_t> adapted_subdoms;
    
    if ( strcmp(meshIn,"box")==0 ) {
        // Count iterations
        totalRefines++;
        printf(" Adaptivity iteration %d\n\n\n",totalRefines); fflush(stdout);
    } else {
        // Read in surfaces and subdomains
        printf(" Reading in the mesh surfaces from %s \n", surfIn);
        MeshFunction<std::size_t> surfReadIn(mesh, surfIn);
        printf(" Reading in the mesh subdomains from %s \n", subdIn);
        MeshFunction<std::size_t> subdReadIn(mesh, subdIn);
        adapted_surfaces = surfReadIn;
        adapted_subdoms  = subdReadIn;

        printf(" Writing mesh\n");
        meshfile << adapted_surfaces;
        meshfile << adapted_subdoms;
    }

    
    
        
    
    //***********************************
    //   Finite Element Space and Forms
    //***********************************
        
    printf("\nDiscretize the Stokes system \n"); fflush(stdout);
    
    // Finite element space
    printf(" Define Stokes stable elements \n"); fflush(stdout);
    linearized_stokes_with_pnp::FunctionSpace VQ(mesh);
    
    // Define variational forms
    printf(" Define Stokes variational forms \n"); fflush(stdout);
    linearized_stokes_with_pnp::BilinearForm a_stokes(VQ, VQ);
    linearized_stokes_with_pnp::LinearForm L_stokes(VQ);


    // Construct mass matrix for pressure (for the uniform preconditioner)
    dCSRmat Mp_bcsr;
    printf("   Constructing the mass matrix for fluidic pressure (for linear solver)\n");
        pressure_mass::FunctionSpace QQ(mesh);
        pressure_mass::BilinearForm M_p(QQ, QQ);
        Matrix MassPress;
        assemble(MassPress,M_p);
        unsigned int mnz = boost::tuples::get<3>(MassPress.data());
        int mrow = MassPress.size(0);
        int mcol = MassPress.size(1);
        int* map = (int*)fasp_mem_calloc(mrow+1, sizeof(int));
        const size_t* map_tmp = boost::tuples::get<0>(MassPress.data());
        for (uint ii=0; ii<mrow+1; ii++) {
            map[ii] = (int)map_tmp[ii];
        }
        int* mai = (int*)fasp_mem_calloc(mnz, sizeof(int));
        const size_t* mai_tmp = boost::tuples::get<1>(MassPress.data());
        for (uint ii=0; ii<mnz; ii++) {
            mai[ii] = (int)mai_tmp[ii];
        }
        double* max = (double*)boost::tuples::get<2>(MassPress.data());
             
        Mp_bcsr.row = mrow;
        Mp_bcsr.col = mcol;
        Mp_bcsr.nnz = mnz;
        Mp_bcsr.IA  = map;
        Mp_bcsr.JA  = mai;
        Mp_bcsr.val = max;
    




    printf("\n\nDiscretize the PNP system \n"); fflush(stdout);
    
    // Finite element space
    printf(" Define PNP finite elements \n"); fflush(stdout);
    linearized_pnp_with_velocity::FunctionSpace W(mesh);
    
    // Define variational forms
    printf(" Define PNP variational forms \n\n"); fflush(stdout);
    linearized_pnp_with_velocity::BilinearForm a(W, W);
    linearized_pnp_with_velocity::LinearForm L(W);
    
    
        
        
    //*********************************************
    //  Mark subdomains and impose Dirichlet B.C.
    //*********************************************
        
    // Define subdomains and subdomains
    printf("Define subdomains \n"); fflush(stdout);
      
      //  PNP subdomains
      dielectricRegion dRegion;
      CellFunction<std::size_t> subdomains_pnp(mesh);
      subdomains_pnp.set_all(1);
      dRegion.mark(subdomains_pnp,2);
      //a.dx = subdomains_pnp;
      //L.dx = subdomains_pnp; 
      //meshfile << subdomains_pnp;

      channelGate channelgate;
      dielectricChannel dchannel;
      FacetFunction<std::size_t> surfaces_pnp(mesh);
      surfaces_pnp.set_all(3);
      channelgate.mark(surfaces_pnp,1);
      dchannel.mark(surfaces_pnp,2);
      L.ds = surfaces_pnp;


    
    // Define Dirichlet boundary conditions
    printf(" Define Dirichlet boundary condition \n\n"); fflush(stdout);
      DirBCval DirBC;
      DirNoFlux noflux(mesh);
      DirichletBoundary inletoutlet;
      ChannelWalls channelwalls;
      SideWalls sides;
      TopAndBottom topbottom;

      FacetFunction<std::size_t> surfaces_stokes(mesh);
      surfaces_stokes.set_all(1);
      channelwalls.mark(surfaces_stokes,2);
      meshfile << surfaces_stokes;


      // Stokes Dirichlet BCs
      SubSpace V(VQ,0);
      DirichletBC bc_stokes_0(V,DirBC,inletoutlet);
      DirichletBC bc_stokes_2(V,noflux,channelwalls);

      //SubSpace V1(V,1);
      //SubSpace V2(V,2);

      //DirichletBC bc_stokes_sides(V1,zero,sides);
      //DirichletBC bc_stokes_topbottom(V2,zero,topbottom);

      std::vector<const DirichletBC*> bc_stokes;
      bc_stokes.push_back(&bc_stokes_0);
      bc_stokes.push_back(&bc_stokes_2);

      // PNP Dirichlet BCs
      DirichletBC bc_1(W,DirBC,inletoutlet);
      DirichletBC bc_2(W,DirBC,inletoutlet);
      //DirichletBC bc_1(W,DirBC,adapted_surfaces,3);
      //DirichletBC bc_2(W,DirBC,adapted_surfaces,4);

      

        
    
    
    //***********************
    // Assign coefficients
    //***********************
    
    printf("Assign coefficients for variational forms \n\n"); fflush(stdout);

    // Stokes eqns
        Constant visc(viscosity);     a_stokes.mu    = visc;  L_stokes.mu    = visc;    // Fluidic viscosity
        Constant alpha(penalty);      a_stokes.alpha = alpha; L_stokes.alpha = alpha;   // DG penalty
        Constant qp(cat_valency);     L_stokes.qp    = qp;                              // Cation valency
        Constant qn(an_valency);      L_stokes.qn    = qn;                              // Anion valency

    // Nernst-Planck eqns
        Constant Dp_p(cat_diff_p);    a.Dp_p = Dp_p; L.Dp_p = Dp_p;  // Cation diffusivity in P-doped
        //Constant Dp_n(cat_diff_n);    a.Dp_n = Dp_n; L.Dp_n = Dp_n;  // Cation diffusivity in N-doped
        Constant Dn_p(an_diff_p);     a.Dn_p = Dn_p; L.Dn_p = Dn_p;  // Anion  diffusivity in P-doped
        //Constant Dn_n(an_diff_n);     a.Dn_n = Dn_n; L.Dn_n = Dn_n;  // Anion  diffusivity in N-doped
        a.qp = qp;   L.qp = qp;       // Cation valency
        a.qn = qn;   L.qn = qn;       // Anion valency
    
    // Poisson eqn
        Constant eps_p(epsPVal);
        //Constant eps_n(epsNVal);
        Constant chrg_p(fix_p);
        Constant chrg_n(fix_n);
        a.eps_p = eps_p; L.eps_p = eps_p;   // P-doped permittivity
        //a.eps_n = eps_n; L.eps_n = eps_n;   // N-doped permittivity
        //L.fix_p = chrg_p;                   // P-doping
        //L.fix_n = chrg_n;                   // N-doping
        Constant pos_surf_charge( 1.0); L.surface_charge1 = pos_surf_charge;
        Constant neg_surf_charge(-1.0); L.surface_charge2 = neg_surf_charge;



    // Linear updates
        L_stokes.du     = zero_vec;
        L_stokes.dPress = zero;
        L_stokes.dPhi   = zero;
        L_stokes.dCat   = zero;
        L_stokes.dAn    = zero;
        L.du   = zero_vec;
        L.dPhi = zero;
        L.dCat = zero;
        L.dAn  = zero;
    
    
    
        
    //***********************************
    //  Interpolate solution iterate
    //***********************************
        
    printf("Define initial guess for Newton iteration \n"); fflush(stdout);
    
    // Interpolate initial guess for Stokes
    Function iterate_stokes(VQ);

        Function VelocIterate(iterate_stokes[0]);
          VelocIterate.interpolate(initVeloc);
        Function PressIterate(iterate_stokes[1]);
          PressIterate.interpolate(initPress);


    // Interpolate initial guess for PNP
    Function iterate(W);
    Function concsoln(W);
    
        Function CatIterate(iterate[0]);
        CatIterate.interpolate(initCat);
        
        Function AnIterate(iterate[1]);
        AnIterate.interpolate(initAn);
        
        // Convert from log to density
        Function CatConc(concsoln[0]);
        Function AnConc(concsoln[1]);
        double * CatConcval = CatConc.vector()->data();
        double * CatItval = CatIterate.vector()->data();
        for(std::size_t i=0; i<CatIterate.vector()->size(); ++i) {
            CatConcval[i] = exp(CatItval[i]);
        }
        double * AnConcval = AnConc.vector()->data();
        double * AnItval = AnIterate.vector()->data();
        for(std::size_t i=0; i<AnIterate.vector()->size(); ++i) {
            AnConcval[i] = exp(AnItval[i]);
        }
    
        // Previous potential
        printf(" Interpolate voltage drop\n"); fflush(stdout);
        Function esIterate(iterate[2]);
        esIterate.interpolate(initPHI);
    
    
    
    
    
    ////////////////////////
    //                    //
    //      Solver        //
    //                    //
    ////////////////////////
        
    printf("\n\n ----------------------------------------\n"); fflush(stdout);
    printf(" Initializing Newton solver \n"); fflush(stdout);
    printf(" ----------------------------------------\n\n"); fflush(stdout);
    
    
    //********************
    //   Solver objects
    //********************
    
    printf("\nInitialize solver objects \n"); fflush(stdout);
    
    // Counters
    int it=0, i;
    double prev_relR, normb_stokes_fasp, normb_fasp;
    double relR_fasp = 1.0;
    bool   done = false;
    
    // Newton updates
    /*Function soln(W);
    Function dNa(soln[0]);
    Function dK(soln[1]);
    Function dCa(soln[2]);
    Function dCl(soln[3]);
    Function dphi(soln[4]);*/
    
    // Block GS updates
    Function solu_stokes(VQ);
    Function solu(W);

    // Newton updates
    Function newton_stokes(VQ);
    Function newton_veloc(newton_stokes[0]);
    Function newton_press(newton_stokes[1]);

    Function newton_pnp(W);
    Function newton_cation(newton_pnp[0]);
    Function newton_anion(newton_pnp[1]);
    Function newton_phi(newton_pnp[2]);
    
    // Linear system
    Matrix A_stokes;
    Vector b_stokes;
        
    Matrix A;
    Vector b;
    
        // Extract DoF indices
        printf(" Assemble solution indices \n"); fflush(stdout);
        std::vector<std::size_t> component(1);
        std::vector<dolfin::la_index> gidx_Cat;
        std::vector<dolfin::la_index> gidx_An;
        std::vector<dolfin::la_index> gidx_phi;
        const dolfin::la_index n0 = W.dofmap()->ownership_range().first;
        const dolfin::la_index n1 = W.dofmap()->ownership_range().second;
        const dolfin::la_index num_dofs = n1 - n0;
        component[0] = 0;
        std::shared_ptr<GenericDofMap> dofmap_Cat  = W.dofmap()->extract_sub_dofmap(component,mesh);
        component[0] = 1;
        std::shared_ptr<GenericDofMap> dofmap_An   = W.dofmap()->extract_sub_dofmap(component,mesh);
        component[0] = 2;
        std::shared_ptr<GenericDofMap> dofmap_phi  = W.dofmap()->extract_sub_dofmap(component,mesh);
        
        for ( CellIterator cell(mesh); !cell.end(); ++cell)
        {
            const std::vector<dolfin::la_index> cell_dofs_Cat = dofmap_Cat->cell_dofs(cell->index());
            const std::vector<dolfin::la_index> cell_dofs_An  = dofmap_An->cell_dofs(cell->index());
            const std::vector<dolfin::la_index> cell_dofs_phi = dofmap_phi->cell_dofs(cell->index());
            for (std::size_t i = 0; i < cell_dofs_Cat.size(); ++i)
            {
                const std::size_t dof = cell_dofs_Cat[i];
                if (dof >= n0 && dof < n1)
                    gidx_Cat.push_back(dof);
            }
            for (std::size_t i = 0; i < cell_dofs_An.size(); ++i)
            {
                const std::size_t dof = cell_dofs_An[i];
                if (dof >= n0 && dof < n1)
                    gidx_An.push_back(dof);
            }
            for (std::size_t i = 0; i < cell_dofs_phi.size(); ++i)
            {
                const std::size_t dof = cell_dofs_phi[i];
                if (dof >= n0 && dof < n1)
                    gidx_phi.push_back(dof);
            }
        }
        std::sort(gidx_Cat.begin(), gidx_Cat.end());
        std::sort(gidx_An.begin(), gidx_An.end());
        std::sort(gidx_phi.begin(), gidx_phi.end());
        // Remove duplicates
        gidx_Cat.erase(std::unique(gidx_Cat.begin(), gidx_Cat.end()), gidx_Cat.end());
        gidx_An.erase(std::unique(gidx_An.begin(),   gidx_An.end()),  gidx_An.end());
        gidx_phi.erase(std::unique(gidx_phi.begin(), gidx_phi.end()), gidx_phi.end());


    //  NS indices
    std::vector<dolfin::la_index> gidx_u;
    std::vector<dolfin::la_index> gidx_press;
    const dolfin::la_index n02 = VQ.dofmap()->ownership_range().first;
    const dolfin::la_index n12 = VQ.dofmap()->ownership_range().second;
    //const dolfin::la_index num_dofs2 = n12 - n02;
    component[0] = 0;
    std::shared_ptr<GenericDofMap> dofmap_u       = VQ.dofmap()->extract_sub_dofmap(component,mesh);
    component[0] = 1;
    std::shared_ptr<GenericDofMap> dofmap_press   = VQ.dofmap()->extract_sub_dofmap(component,mesh);
        
    for (CellIterator cell2(mesh); !cell2.end(); ++cell2) {
        const std::vector<dolfin::la_index> cell_dofs_u     = dofmap_u->cell_dofs(cell2->index());
        const std::vector<dolfin::la_index> cell_dofs_press = dofmap_press->cell_dofs(cell2->index());
            
        for (std::size_t i = 0; i < cell_dofs_u.size(); ++i)
        {
            const std::size_t dof = cell_dofs_u[i];
            if (dof >= n02 && dof < n12)
                gidx_u.push_back(dof);
        }
        for (std::size_t i = 0; i < cell_dofs_press.size(); ++i)
        {
            const std::size_t dof = cell_dofs_press[i];
            if (dof >= n02 && dof < n12)
                gidx_press.push_back(dof);
        }
    }
    std::sort(gidx_u.begin(),     gidx_u.end());
    std::sort(gidx_press.begin(), gidx_press.end());
        
    // Remove duplicates
    gidx_u.erase(std::unique(gidx_u.begin(), gidx_u.end()), gidx_u.end());
    gidx_press.erase(std::unique(gidx_press.begin(), gidx_press.end()), gidx_press.end());

        
    // write index to file
    /*
     
     printf(" Write index vectors to files \n"); fflush(stdout);
     FILE* pidxfile;
     pidxfile = fopen("pidx.dat", "w");
     fprintf(pidxfile, "%d \n",gidx_p.size());
     for(std::size_t i=0; i<gidx_p.size(); i++)
     fprintf(pidxfile, "%d \n",gidx_p[i]);
     fclose(pidxfile);
     
     FILE* nidxfile;
     nidxfile = fopen("nidx.dat", "w");
     fprintf(nidxfile, "%d \n",gidx_n.size());
     for(std::size_t i=0; i<gidx_n.size(); i++)
     fprintf(nidxfile, "%d \n",gidx_n.data()[i]);
     fclose(nidxfile);
     
     FILE* phiidxfile;
     phiidxfile = fopen("phiidx.dat", "w");
     fprintf(phiidxfile, "%d \n",gidx_phi.size());
     for(std::size_t i=0; i<gidx_phi.size(); i++)
     fprintf(phiidxfile, "%d \n",gidx_phi.data()[i]);
     fclose(phiidxfile);
     
     
     //  Write RHS to file
     printf(" Write RHS to file  \n"); fflush(stdout);
     dvector b_fasp;
     b_fasp.row = b.size();
     b_fasp.val = (double*)b.data();
     
     FILE* bfile;
     bfile = fopen("rhs.dat", "w");
     fprintf(bfile, "%d \n",b_fasp.row);
     for( i=0; i<b_fasp.row; i++)
     fprintf(bfile, "%f \n",b_fasp.val[i]);
     fclose(bfile);
     */





    
    // Estimate the initial residual
    printf(" Measure initial residual \n");

        // Stokes residual
        L_stokes.uu     = VelocIterate;
        L_stokes.pp     = PressIterate;
        L_stokes.phi    = esIterate;
        L_stokes.cation = CatIterate;
        L_stokes.anion  = AnIterate;
        assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
        bc_stokes_2.apply(b_stokes);
        //bc_stokes_sides.apply(b_stokes); bc_stokes_topbottom.apply(b_stokes);

        dvector b_stokes_fasp;
        b_stokes_fasp.row = b_stokes.size();
        b_stokes_fasp.val = (double*)b_stokes.data();
        if (b0_stokes_fasp < 0.) {   // Initial residual
            b0_stokes_fasp = fasp_blas_array_norm2(b_stokes_fasp.row,b_stokes_fasp.val);
            printf(" The initial Stokes residual is %e \n\n", b0_stokes_fasp);
        }




        //  PNP Residual
        L.uu     = VelocIterate;
        L.CatCat = CatIterate;
        L.AnAn   = AnIterate;
        L.EsEs   = esIterate;
        assemble(b,L); bc_1.apply(b); bc_2.apply(b);
        //bc_membrane_0.apply(b); bc_membrane_1.apply(b); bc_membrane_2.apply(b); bc_membrane_3.apply(b); bc_protein_0.apply(b);
        //bc_protein_1.apply(b); bc_protein_2.apply(b); bc_protein_3.apply(b);
    
        dvector b_fasp;
        b_fasp.row = b.size();
        b_fasp.val = (double*)b.data();
        if (b0_fasp < 0.) {   // Initial residual
            b0_fasp = fasp_blas_array_norm2(b_fasp.row,b_fasp.val);
            printf(" The initial PNP residual is %e \n\n", b0_fasp);
        }

    double initial_residual = sqrt(b0_stokes_fasp*b0_stokes_fasp + b0_fasp*b0_fasp);
    printf(" The combined initial residual is %e \n\n", initial_residual); fflush(stdout);


    
    // Start Timer
	timeval tim ;
	gettimeofday(&tim, NULL) ;
	double runtime = tim.tv_sec+(tim.tv_usec/1000000.0) ;
    
    
    //********************
    //  Newton iteration
    //********************
    
    while (done==false) {
        
        // Save solution in VTK format
        printf(" Write solution to file \n\n"); fflush(stdout);
        Velocityfile << VelocIterate;
        Pressurefile << PressIterate;

        Catfile  << CatConc;
        Anfile   << AnConc;
        phifile  << esIterate;
        
        
        
        // Update newton step
        it++;
        printf("Newton iteration %d\n", it); fflush(stdout);
        printf(" Construct Jacobian matrix \n"); fflush(stdout);

        // Update Stokes coefficients
        //a_stokes.uu = VelocIterate;
        assemble(A_stokes,a_stokes); bc_stokes_0.apply(A_stokes); 
        bc_stokes_2.apply(A_stokes);
        //bc_stokes_sides.apply(A_stokes); bc_stokes_topbottom.apply(A_stokes);
        
        // Update PNP coefficients
        a.uu     = VelocIterate;
        a.CatCat = CatIterate;
        a.AnAn   = AnIterate;
        a.EsEs   = esIterate;
        assemble(A,a); bc_1.apply(A); bc_2.apply(A);
        //bc_membrane_0.apply(A); bc_membrane_1.apply(A); bc_membrane_2.apply(A); bc_membrane_3.apply(A);
        //bc_protein_0.apply(A); bc_protein_1.apply(A); bc_protein_2.apply(A); bc_protein_3.apply(A);



        // ********************
        // Block Gauss-Seidel
        // ********************
        printf(" Convert linearized PDE to FASP structures \n\n"); fflush(stdout);


        printf(" ----------------------------------------------\n"); fflush(stdout);
        printf(" Start conversion to FASP for Stokes subsystem \n"); fflush(stdout);
        printf(" ----------------------------------------------\n\n"); fflush(stdout);
        
            printf(" Step 1: convert sparse matrix format\n"); fflush(stdout);
            // Convert A2 to CSR
            dCSRmat A_stokes_fasp;
            block_dCSRmat A_stokes_bcsr;
            //dCSRmat MP;
            
            // get index
            INT nrow_u     = gidx_u.size();
            INT nrow_press = gidx_press.size();
            
            ivector u_idx;
            ivector press_idx;
            
            fasp_ivec_alloc(nrow_u, &u_idx);
            fasp_ivec_alloc(nrow_press, &press_idx);
            
            
            for(std::size_t jj=0; jj<nrow_u; jj++)
                u_idx.val[jj] = gidx_u[jj];
            for(std::size_t jj=0; jj<nrow_press; jj++)
                press_idx.val[jj] = gidx_press[jj];
            
            // Assign A_stokes_fasp
            int ii;
            unsigned int nz2 = boost::tuples::get<3>(A_stokes.data());
            int row2 = A_stokes.size(0);
            int col2 = A_stokes.size(1);
            int* ap2 = (int*)fasp_mem_calloc(row2+1, sizeof(int));
            const size_t* ap_tmp2 = boost::tuples::get<0>(A_stokes.data());
            
            for (ii=0; ii<row2+1; ii++) {
                ap2[ii] = (int)ap_tmp2[ii];
            }
            
            int* ai2 = (int*)fasp_mem_calloc(nz2, sizeof(int));
            const size_t* ai_tmp2 = boost::tuples::get<1>(A_stokes.data());
            for (ii=0; ii<nz2; ii++) {
                ai2[ii] = (int)ai_tmp2[ii];
            }
            
            double* ax2 = (double*)boost::tuples::get<2>(A_stokes.data());
            
            A_stokes_fasp.row = row2;
            A_stokes_fasp.col = col2;
            A_stokes_fasp.nnz = nz2;
            A_stokes_fasp.IA  = ap2;
            A_stokes_fasp.JA  = ai2;
            A_stokes_fasp.val = ax2;
            
            // extract blocks
            A_stokes_bcsr.brow = 2;
            A_stokes_bcsr.bcol = 2;
            A_stokes_bcsr.blocks = (dCSRmat **)calloc(4, sizeof(dCSRmat *));
            fasp_mem_check((void *)A_stokes_bcsr.blocks, "block matrix:cannot allocate memory!\n", ERROR_ALLOC_MEM);
            for (i=0; i<4 ;i++) {
                A_stokes_bcsr.blocks[i] = (dCSRmat *)fasp_mem_calloc(1, sizeof(dCSRmat));
            }
            
            // get Auu block
            fasp_dcsr_getblk(&A_stokes_fasp, u_idx.val,     u_idx.val,     u_idx.row,     u_idx.row,     A_stokes_bcsr.blocks[0]);
            // get Aup block
            fasp_dcsr_getblk(&A_stokes_fasp, u_idx.val,     press_idx.val, u_idx.row,     press_idx.row, A_stokes_bcsr.blocks[1]);
            // get Apu block
            fasp_dcsr_getblk(&A_stokes_fasp, press_idx.val, u_idx.val,     press_idx.row, u_idx.row,     A_stokes_bcsr.blocks[2]);
            // get App block
            fasp_dcsr_getblk(&A_stokes_fasp, press_idx.val, press_idx.val, press_idx.row, press_idx.row, A_stokes_bcsr.blocks[3]);
            




            dvector b_stokes_bcsr;
            fasp_dvec_alloc(b_stokes_fasp.row, &b_stokes_bcsr);
            for (i=0; i<nrow_u; i++)
                b_stokes_bcsr.val[i]        = b_stokes_fasp.val[gidx_u[i]];
            for (i=0; i<nrow_press; i++)
                b_stokes_bcsr.val[nrow_u+i] = b_stokes_fasp.val[gidx_press[i]];
            
            dvector soluvec_stokes;
            fasp_dvec_alloc(b_stokes_fasp.row, &soluvec_stokes);
            fasp_dvec_set(b_stokes_fasp.row, &soluvec_stokes, 0.0);



            printf(" Step 2: initialize solver parameters\n"); fflush(stdout);
            char inputfile2[] = "./params/nsbcsr.dat";
            input_ns_param     inparam; // parameters from input files
            itsolver_ns_param  itparam; // parameters for itsolver
            AMG_ns_param      amgparam; // parameters for AMG
            ILU_param         iluparam; // parameters for ILU
            Schwarz_param     schparam; // parameters for Schwarz
            
            fasp_ns_param_input(inputfile2,&inparam);
            fasp_ns_param_init(&inparam, &itparam, &amgparam, &iluparam, &schparam);





        
        // **************************************************
        // Interface with FASP for PNP
        // **************************************************
        
        printf(" ------------------------------------------\n"); fflush(stdout);
        printf(" Start conversion to FASP for PNP subsystem \n"); fflush(stdout);
        printf(" ------------------------------------------\n\n"); fflush(stdout);
        
        printf(" Step 1: convert sparse matrix format\n"); fflush(stdout);
        // Convert A to CSR
        dCSRmat A_fasp;
        
        unsigned int nz = boost::tuples::get<3>(A.data());
        int row = A.size(0);
        int col = A.size(1);
        int* ap = (int*)fasp_mem_calloc(row+1, sizeof(int));
        const size_t* ap_tmp = boost::tuples::get<0>(A.data());
        for (i=0; i<row+1; i++) {
            ap[i] = (int)ap_tmp[i];
        }
        int* ai = (int*)fasp_mem_calloc(nz, sizeof(int));
        const size_t* ai_tmp = boost::tuples::get<1>(A.data());
        for (i=0; i<nz; i++) {
            ai[i] = (int)ai_tmp[i];
        }
        double* ax = (double*)boost::tuples::get<2>(A.data());







        
        // Check for rows of zeros and add a unit diagonal entry
        bool nonzero_entry = false;
        for ( uint rowInd=0; rowInd<row; rowInd++ ) {
            
            // Check for nonzero entry
            nonzero_entry   = false;
            int diagColInd = -1;
            if ( ap[rowInd] < ap[rowInd+1] ) {
                for ( uint colInd=ap[rowInd]; colInd < ap[rowInd+1]; colInd++ ) {
                    if ( ax[colInd] != 0.0 )    nonzero_entry = true;
                    if ( ai[colInd] == rowInd ) diagColInd    = colInd;    // marks diagonal entry
                }
            }
            
            
            if ( diagColInd < 0 ) {
                printf(" ERROR: diagonal entry not allocated!!\n\n Exiting... \n \n"); fflush(stdout);
                printf("      for row %d\n",rowInd); fflush(stdout);
                for ( uint colInd=ap[rowInd]; colInd < ap[rowInd+1]; colInd++ ) {
                    printf("          %d\n",ai[colInd]);
                }
                return 0;
            }
            
            if ( nonzero_entry==false ) {
                //printf(" Row %d has only zeros! Setting diagonal entry to 1.0 \n", rowInd); fflush(stdout);
                ax[diagColInd] = 1.0;
                //b_fasp.val[rowInd] = -1.e+5;
            }
        }

        A_fasp.row = row;
        A_fasp.col = col;
        A_fasp.nnz = nz;
        A_fasp.IA  = ap;
        A_fasp.JA  = ai;
        A_fasp.val = ax;
        
#if FASP_BSR
        // convert CSR to BSR
        dBSRmat A_fasp_bsr = fasp_format_dcsr_dbsr(&A_fasp, 3);
        
        // output
        /*
         fasp_dbsr_write("A_fasp_bsr.dat", &A_fasp_bsr);
         fasp_dvec_write("b_fasp.dat", &b_fasp);
         */
#else
        // convert CSR to block CSR
        block_dCSRmat Abcsr;
        dCSRmat *A_diag;
        
        // get index
        INT nrow = A_fasp.row/3;
        ivector phi_idx;
        ivector An_idx;
        ivector Cat_idx;
        
        fasp_ivec_alloc(nrow, &phi_idx);
        fasp_ivec_alloc(nrow, &An_idx);
        fasp_ivec_alloc(nrow, &Cat_idx);
        
        for (i=0; i<nrow; i++){
            phi_idx.val[i] = 3*i;
            An_idx.val[i]  = 3*i+1;
            Cat_idx.val[i] = 3*i+2;
        }
        
        // Assemble the matrix in block dCSR format
        Abcsr.brow = 3; Abcsr.bcol = 3;
        Abcsr.blocks = (dCSRmat **)calloc(9, sizeof(dCSRmat *));
        for (i=0; i<9 ;i++) {
            Abcsr.blocks[i] = (dCSRmat *)fasp_mem_calloc(1, sizeof(dCSRmat));
        }
        
        // A11
        fasp_dcsr_getblk(&A_fasp, phi_idx.val, phi_idx.val, nrow, nrow, Abcsr.blocks[0]);
        // A12
        fasp_dcsr_getblk(&A_fasp, phi_idx.val, An_idx.val, nrow, nrow, Abcsr.blocks[1]);
        // A12
        fasp_dcsr_getblk(&A_fasp, phi_idx.val, Cat_idx.val, nrow, nrow, Abcsr.blocks[2]);
        
        // A21
        fasp_dcsr_getblk(&A_fasp, An_idx.val, phi_idx.val, nrow, nrow, Abcsr.blocks[3]);
        // A22
        fasp_dcsr_getblk(&A_fasp, An_idx.val, An_idx.val, nrow, nrow, Abcsr.blocks[4]);
        // A23
        fasp_dcsr_getblk(&A_fasp, An_idx.val, Cat_idx.val, nrow, nrow, Abcsr.blocks[5]);
        
        // A31
        fasp_dcsr_getblk(&A_fasp, Cat_idx.val, phi_idx.val, nrow, nrow, Abcsr.blocks[6]);
        // A32
        fasp_dcsr_getblk(&A_fasp, Cat_idx.val, An_idx.val, nrow, nrow, Abcsr.blocks[7]);
        // A33
        fasp_dcsr_getblk(&A_fasp, Cat_idx.val, Cat_idx.val, nrow, nrow, Abcsr.blocks[8]);
        
        
        // setup diagonal blocks for the preconditioner
        A_diag = (dCSRmat *)fasp_mem_calloc(3, sizeof(dCSRmat));
        
        // first diagonal block
        A_diag[0].row = Abcsr.blocks[0]->row;
        A_diag[0].col = Abcsr.blocks[0]->col;
        A_diag[0].nnz = Abcsr.blocks[0]->nnz;
        A_diag[0].IA  = Abcsr.blocks[0]->IA;
        A_diag[0].JA  = Abcsr.blocks[0]->JA;
        A_diag[0].val = Abcsr.blocks[0]->val;
        
        // second diagonal block
        A_diag[1].row = Abcsr.blocks[4]->row;
        A_diag[1].col = Abcsr.blocks[4]->col;
        A_diag[1].nnz = Abcsr.blocks[4]->nnz;
        A_diag[1].IA  = Abcsr.blocks[4]->IA;
        A_diag[1].JA  = Abcsr.blocks[4]->JA;
        A_diag[1].val = Abcsr.blocks[4]->val;
        
        // third diagonal block
        A_diag[2].row = Abcsr.blocks[8]->row;
        A_diag[2].col = Abcsr.blocks[8]->col;
        A_diag[2].nnz = Abcsr.blocks[8]->nnz;
        A_diag[2].IA  = Abcsr.blocks[8]->IA;
        A_diag[2].JA  = Abcsr.blocks[8]->JA;
        A_diag[2].val = Abcsr.blocks[8]->val;

        //fasp_dcoo_write("aDiag0",&A_diag[0]);
        //fasp_dcoo_write("aDiag1",&A_diag[1]);
        //fasp_dcoo_write("aDiag2",&A_diag[2]);
        
        
        
        // convert right hand side
        dvector bbcsr;
        fasp_dvec_alloc(b_fasp.row, &bbcsr);
        for (i=0; i<nrow; i++){
            bbcsr.val[i]        = b_fasp.val[3*i];
            bbcsr.val[nrow+i]   = b_fasp.val[3*i+1];
            bbcsr.val[2*nrow+i] = b_fasp.val[3*i+2];
        }
        
        // output the matrices
        /*
         fasp_dcoo_write("A.dat", &A_fasp);
         fasp_ivec_write("nidx.dat", &n_idx);
         fasp_ivec_write("phiidx.dat", &phi_idx);
         fasp_ivec_write("pidx.dat", &p_idx);
         fasp_dvec_write("rhs.dat", &bbcsr);
         
         getchar();
         */
        
#endif
        
        
        // free CSR matrix
        //fasp_dcsr_free(&A_fasp);
        
        // initialize solution
        dvector soluvec;
        fasp_dvec_alloc(b_fasp.row, &soluvec);
        fasp_dvec_set(b_fasp.row, &soluvec, 0.0);
        
        printf(" Step 2: initialize solver parameters\n"); fflush(stdout);
        // initialize solver parameters
        input_param     inpar;  // parameters from input files
        itsolver_param  itpar;  // parameters for itsolver
        AMG_param       amgpar; // parameters for AMG
        ILU_param       ilupar; // parameters for ILU
        
        // read in parameters from a input file
#if FASP_BSR
        char inputfile[] = "./params/bsr.dat";
#else
        char inputfile[] = "./params/bcsr.dat";
#endif
        fasp_param_input(inputfile, &inpar);
        fasp_param_init(&inpar, &itpar, &amgpar, &ilupar, NULL);








        // *************************************
        // Block Gauss-Seidel solver using FASP
        // *************************************
        printf("\n\n Execute Gauss-Seidel sweeps on the coupled Jacobian Matrix \n"); fflush(stdout);



        // Initialize updates from linearized systems (for constructing linearized residuals)
        Function linear_stokes(VQ);
        Function linear_veloc(linear_stokes[0]); newton_veloc.interpolate(zero_vec);
        Function linear_press(linear_stokes[1]); newton_press.interpolate(zero);

        Function linear_pnp(W);
        Function linear_cation(linear_pnp[0]);  newton_cation.interpolate(zero);
        Function linear_anion(linear_pnp[1]);   newton_anion.interpolate(zero);
        Function linear_phi(linear_pnp[2]);     newton_phi.interpolate(zero);
            
        // Initial linearized residual
        uint   bGSit = 0;
        double lin_normb_fasp0  = fasp_blas_array_norm2(b_fasp.row,b_fasp.val);
        double lin_normb_stokes_fasp0 = fasp_blas_array_norm2(b_stokes_fasp.row,b_stokes_fasp.val);
        double lin_normb  = lin_normb_fasp0;
        double lin_normb_stokes = lin_normb_stokes_fasp0;
        double prev_lin_relR, lin_relR = 1.0;
        double initial_lin_res = sqrt(lin_normb_stokes*lin_normb_stokes + lin_normb*lin_normb);
        bool   linear_convergence = false;





        // DEBUG: check frobenius norms
        double A_stokes_frob = 0.0;
        for(uint frobind = 0; frobind < nz2; frobind++) {
            A_stokes_frob += ax2[frobind]*ax2[frobind];
        }

        printf("    Matrix norm: || A_stokes ||_F^2 = %e \n",A_stokes_frob);fflush(stdout);

#if FASP_NS_MASS
        double M_p_frob = 0.0;
        for(uint frobind = 0; frobind < mnz; frobind++) {
            M_p_frob += Mp_bcsr.val[frobind]*Mp_bcsr.val[frobind];
        }

        printf("    Matrix norm: || M_p ||_F^2      = %e \n",M_p_frob); fflush(stdout);
#endif
        printf("    Vector norm: || b_stokes ||_0^2 = %e \n",lin_normb_stokes_fasp0);fflush(stdout);

        double A_frob = 0.0;
        for(uint frobind = 0; frobind < nz; frobind++) {
            A_frob += ax[frobind]*ax[frobind];
        }
        printf("    Matrix norm: || A ||_F^2 = %e \n",A_frob);fflush(stdout);
        printf("    Vector norm: || b ||_0^2 = %e \n",lin_normb_fasp0);fflush(stdout);



            
        while ( !linear_convergence ) {

        // Update iteration count
        bGSit++;

        //********************
        //   Stokes Solve
        //********************
                
        printf("\n Step 3.1: solve the Stokes' linearized system\n"); fflush(stdout);
        INT flag = 0;
        fasp_dvec_set(b_stokes_fasp.row, &soluvec_stokes, 0.0); // start with initial guess zero
        //flag = fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass(&A2bcsr, &b2bcsr, &NSsoluvec, &itparam, &amgparam, &iluparam, &schparam, &MP);

#if FASP_NS_MASS
        flag = fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass(&A_stokes_bcsr, &b_stokes_bcsr, &soluvec_stokes, &itparam, &amgparam, &iluparam, &schparam, &Mp_bcsr);
#else        
        flag = fasp_solver_bdcsr_krylov_navier_stokes(&A_stokes_bcsr, &b_stokes_bcsr, &soluvec_stokes, &itparam, &amgparam, &iluparam, &schparam);
#endif
        if (flag<0) {
            printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", flag); fflush(stdout);
        }
        else {
            printf("\nSolver finished successfully!\n\n"); fflush(stdout);
        }


        // Convert solution vector to FE solution
        printf(" Step 3.2: convert Stokes' solution to FEniCS structure\n"); fflush(stdout);

        // Convert Stokes solution
        double * NSsolval = solu_stokes.vector()->data();
        for (std::size_t i=0; i<nrow_u; ++i)
            NSsolval[gidx_u[i]] = soluvec_stokes.val[i];
        for (std::size_t i=0; i<nrow_press; ++i)
            NSsolval[gidx_press[i]]   = soluvec_stokes.val[nrow_u+i];

        // Add to Stokes nonlinear update
        linear_veloc.interpolate(solu_stokes[0]); *(newton_veloc.vector()) += *(linear_veloc.vector());
        linear_press.interpolate(solu_stokes[1]); *(newton_press.vector()) += *(linear_press.vector());

        // Update linearized residuals and convert
        L.du            = newton_veloc;
        L_stokes.du     = newton_veloc;
        L_stokes.dPress = newton_press;
        assemble(b,L); bc_1.apply(b); bc_2.apply(b);
        b_fasp.val = (double*)b.data();
#if FASP_BSR
        printf(" Residual for PNP already in FASP format \n"); fflush(stdout);
#else
        for (i=0; i<nrow; i++){
            bbcsr.val[i]        = b_fasp.val[3*i];
            bbcsr.val[nrow+i]   = b_fasp.val[3*i+1];
            bbcsr.val[2*nrow+i] = b_fasp.val[3*i+2];
        }
#endif

        // Calculate residual for stokes (to verify agreement with solver)
        assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
        bc_stokes_2.apply(b_stokes);
        //bc_stokes_sides.apply(b_stokes); bc_stokes_topbottom.apply(b_stokes);
            b_stokes_fasp.val = (double*) b_stokes.data();
            for (i=0; i<nrow_u; i++)
                b_stokes_bcsr.val[i]        = b_stokes_fasp.val[gidx_u[i]];
            for (i=0; i<nrow_press; i++)
                b_stokes_bcsr.val[nrow_u+i] = b_stokes_fasp.val[gidx_press[i]];
        lin_normb_stokes = fasp_blas_array_norm2(b_stokes_fasp.row,b_stokes_fasp.val);
        printf(" The linearized Stokes' residual is %e \n", lin_normb_stokes);







        printf(" Step 4.1: solve the PNP linearized system\n"); fflush(stdout);
        // solve
        int status=FASP_SUCCESS;
        fasp_dvec_set(b_fasp.row, &soluvec, 0.0); // initial guess is zero
        //fasp_param_amg_print(&amgpar);
        
#if FASP_BSR
        status = fasp_solver_dbsr_krylov_amg(&A_fasp_bsr, &b_fasp, &soluvec, &itpar, &amgpar);
#else
        status = fasp_solver_bdcsr_krylov_block_3(&Abcsr, &bbcsr, &soluvec, &itpar, &amgpar, A_diag);
#endif
        
        if (status<0) {
            printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", status); fflush(stdout);
        }
        else {
            printf("\nSolver finished successfully!\n\n"); fflush(stdout);
        }

        
        printf(" Step 3.2: convert PNP solution to FEniCS structure\n"); fflush(stdout);
        // Convert PNP solution
        double * solval = solu.vector()->data();
#if FASP_BSR
        for(std::size_t i=0; i<soluvec.row; ++i) {
            solval[i] = soluvec.val[i];
        }
#else
        for(std::size_t i=0; i<nrow; ++i) {
            solval[3*i]    = soluvec.val[i];
            solval[3*i+1]  = soluvec.val[nrow+i];
            solval[3*i+2]  = soluvec.val[2*nrow+i];
        }
#endif
        // Add to PNP nonlinear update
        linear_cation.interpolate(solu[0]); *(newton_cation.vector()) += *(linear_cation.vector());
        linear_anion.interpolate(solu[1]);  *(newton_anion.vector())  += *(linear_anion.vector());
        linear_phi.interpolate(solu[2]);    *(newton_phi.vector())    += *(linear_phi.vector());


        // Stokes residual and conversion
        L_stokes.dCat = newton_cation;
        L_stokes.dAn  = newton_anion;
        L_stokes.dPhi = newton_phi;

            assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
            bc_stokes_2.apply(b_stokes);
            //bc_stokes_sides.apply(b_stokes); bc_stokes_topbottom.apply(b_stokes);
            b_stokes_fasp.val = (double*) b_stokes.data();
            for (i=0; i<nrow_u; i++)
                b_stokes_bcsr.val[i]        = b_stokes_fasp.val[gidx_u[i]];
            for (i=0; i<nrow_press; i++)
                b_stokes_bcsr.val[nrow_u+i] = b_stokes_fasp.val[gidx_press[i]];


        // PNP residual and conversion
        L.dCat = newton_cation;
        L.dAn  = newton_anion;
        L.dPhi = newton_phi;

            assemble(b,L); bc_1.apply(b); bc_2.apply(b);
            b_fasp.val = (double*)b.data();
#if FASP_BSR
            printf(" Residual for PNP already in FASP format \n"); fflush(stdout);
#else
            for (i=0; i<nrow; i++){
                bbcsr.val[i]        = b_fasp.val[3*i];
                bbcsr.val[nrow+i]   = b_fasp.val[3*i+1];
                bbcsr.val[2*nrow+i] = b_fasp.val[3*i+2];
            }
#endif

        // Compute linearized relative residuals
        lin_normb        = fasp_blas_array_norm2(b_fasp.row,b_fasp.val);
        lin_normb_stokes = fasp_blas_array_norm2(b_stokes_fasp.row,b_stokes_fasp.val);
        printf(" The linearized PNP residual is %e \n", lin_normb);
        printf(" The linearized Stokes residual is %e \n", lin_normb_stokes);

        prev_lin_relR = lin_relR;
        lin_relR      = sqrt(lin_normb_stokes*lin_normb_stokes + lin_normb*lin_normb) / initial_lin_res;
        printf(" The combined linearized relative residual is %e after %d block-GS iteration(s) \n\n", lin_relR, bGSit);
        fflush(stdout);

        if ( lin_relR < bGS_tol ) {
            printf(" Block-GS scheme converged \n\n"); fflush(stdout);
            linear_convergence = true;
        } else if ( bGSit >= max_bGS_it ) {
            printf(" *** Block-GS scheme did not converge! *** \n\n"); fflush(stdout);
            linear_convergence = true;
        }


            // DEBUG: write out linear updates
            /*printf(" Write solution to file \n"); fflush(stdout);
            Velocityfile << newton_veloc;
            Pressurefile << newton_press;

            Catfile << newton_cation;
            Anfile  << newton_anion;
            phifile << newton_phi;*/


    }




        
        printf(" Step 5: free memory\n"); fflush(stdout);
        // Free memory
#if FASP_BSR
        fasp_dbsr_free(&A_fasp_bsr);
#else
        fasp_bdcsr_free(&Abcsr);
        //for (std::size_t i=0; i<9; i++) fasp_dcsr_free(Abcsr.blocks[i]);
        //fasp_mem_free(&Abcsr);
        fasp_dvec_free(&bbcsr);
        fasp_ivec_free(&phi_idx);
        fasp_ivec_free(&An_idx);
        fasp_ivec_free(&Cat_idx);
#endif
        //fasp_dvec_free(&b_fasp);
        fasp_dvec_free(&soluvec);
        free(ap); free(ai);


        // Free memory
        fasp_bdcsr_free(&A_stokes_bcsr);
        fasp_dvec_free(&b_stokes_bcsr);
        fasp_ivec_free(&u_idx);
        fasp_ivec_free(&press_idx);
        fasp_dvec_free(&soluvec_stokes);
        free(ap2); free(ai2);


        printf(" ---------------------------\n"); fflush(stdout);
        printf(" End of interface to FASP \n"); fflush(stdout);
        printf(" ---------------------------\n\n"); fflush(stdout);

        
        //  Nullify linearized terms in residuals
        L_stokes.du     = zero_vec;
        L_stokes.dPress = zero;
        L_stokes.dCat   = zero;
        L_stokes.dAn    = zero;
        L_stokes.dPhi   = zero;
        L.du   = zero_vec;
        L.dCat = zero;
        L.dAn  = zero;
        L.dPhi = zero;



        









        //***************************
        //      Update solution
        //***************************
        
        printf(" Update solution \n"); fflush(stdout);

        // Stokes Update
        //Function stokes_update(VQ);
        Function dVeloc(newton_veloc);  //   updateVEL.interpolate(solu_stokes[0]);
        Function dPress(newton_press);// updatePRESS.interpolate(solu_stokes[1]);


        // PNP Update
        //Function NewtonUpdate(W);
        Function dCat(newton_cation);
        Function dAn(newton_anion);
        Function dphi(newton_phi);
        //Function dCat(NewtonUpdate[0]); dCat.interpolate(solu[0]);
        //Function dAn(NewtonUpdate[1]);   dAn.interpolate(solu[1]);
        //Function dphi(NewtonUpdate[2]); dphi.interpolate(solu[2]);
        
        
        //  Backtrack line search
        printf("   Use backtracking to guarantee decreasing residual \n"); fflush(stdout);
        // stokes
        Function update_stokes(VQ);
        Function updateVELOC(update_stokes[0]);
        Function updatePRESS(update_stokes[1]);

        // pnp
        Function update(W);
        Function updateCAT(update[0]);
        Function updateAN(update[1]);
        Function updatePHI(update[2]);
        double test_stokes, test_pnp;
        double testRelRes  = relR_fasp+DOLFIN_EPS;//1.0+DOLFIN_EPS;
        double dampFactor  = mu;
        bool   reducing  = true;   // loop boolean
        bool   reduced   = false;  // residual is reduced?
        bool   tooDamped = false;  // dampFactor too small?
        while ( reducing ) {//testRelRes > relR_fasp-DOLFIN_EPS && dampFactor > 1.e-4 ) {
            
            //  Compute updates
            updateVELOC.interpolate(VelocIterate);
            updatePRESS.interpolate(PressIterate);
            *(updateVELOC.vector()) += *(dVeloc.vector());
            *(updatePRESS.vector()) += *(dPress.vector());

            updateCAT.interpolate(CatIterate);
            updateAN.interpolate(AnIterate);
            updatePHI.interpolate(esIterate);
            *(updateCAT.vector())  += *(dCat.vector());
            *(updateAN.vector())   += *(dAn.vector());
            *(updatePHI.vector())  += *(dphi.vector());
            
            // Evaluate residual
            L_stokes.uu      = updateVELOC;
            L_stokes.pp      = updatePRESS;
            L_stokes.cation  = updateCAT;
            L_stokes.anion   = updateAN;
            L_stokes.phi     = updatePHI;
            assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
            bc_stokes_2.apply(b_stokes);
            //bc_stokes_sides.apply(b_stokes); bc_stokes_topbottom.apply(b_stokes);

            L.uu     = updateVELOC;
            L.CatCat = updateCAT;
            L.AnAn   = updateAN;
            L.EsEs   = updatePHI;
            assemble(b,L); bc_1.apply(b); bc_2.apply(b);
            //bc_membrane_0.apply(b); bc_membrane_1.apply(b); bc_membrane_2.apply(b); bc_membrane_3.apply(b);
            //bc_protein_0.apply(b); bc_protein_1.apply(b); bc_protein_2.apply(b); bc_protein_3.apply(b);
            
            // Compute relative residual for backtrack line search
            // Stokes
            dvector newb_stokes_faspBT;
            newb_stokes_faspBT.row = b_stokes.size();
            newb_stokes_faspBT.val = (double*)b_stokes.data();
            double normb_stokes_faspBT = fasp_blas_array_norm2(newb_stokes_faspBT.row,newb_stokes_faspBT.val);
            test_stokes  = normb_stokes_faspBT/b0_stokes_fasp ;
            printf("   Stokes' residual after update is %e \n", test_stokes);
            fflush(stdout);

            // PNP
            dvector newb_faspBT;
            newb_faspBT.row = b.size();
            newb_faspBT.val = (double*)b.data();
            double normb_faspBT = fasp_blas_array_norm2(newb_faspBT.row,newb_faspBT.val);
            test_pnp   = normb_faspBT/b0_fasp ;
            printf("   PNP residual after update is %e \n", test_pnp);
            fflush(stdout);


            // Combined
            testRelRes = sqrt(normb_stokes_faspBT*normb_stokes_faspBT + normb_faspBT*normb_faspBT)/initial_residual;
            printf("   The combined residual after update is %e \n", testRelRes);
            fflush(stdout);
            
            // check for reduced residual or NaN
            if ( (testRelRes>relR_fasp-DOLFIN_EPS) || (testRelRes!=testRelRes) ) {
                dampFactor            *= 0.5;
                printf("   Reducing the update by a factor of %e \n", dampFactor); fflush(stdout);

                *(dVeloc.vector())    *= dampFactor;
                *(dPress.vector())    *= dampFactor;

                *(dCat.vector())      *= dampFactor;
                *(dAn.vector())       *= dampFactor;
                *(dphi.vector())      *= dampFactor;

            } else {// Significant reduction
                reduced = true;
            }
            if (dampFactor<1.0e+0)   tooDamped = true;  // too much backtracking
            if (reduced || tooDamped) reducing = false; // end while
        }
        
        // If backtrack failed
        if ( testRelRes > relR_fasp-DOLFIN_EPS ){
            printf("   BAD UPDATE!!! \n"); fflush(stdout);
        }

        
        // Update solutions
        *(VelocIterate.vector()) = *(updateVELOC.vector());
        *(PressIterate.vector()) = *(updatePRESS.vector());

        *(CatIterate.vector()) = *(updateCAT.vector());
        *(AnIterate.vector())  = *(updateAN.vector());
        *(esIterate.vector())  = *(updatePHI.vector());
        


        // Update residual
        printf(" Update variational forms \n"); fflush(stdout);
        L_stokes.uu      = VelocIterate;
        L_stokes.pp      = PressIterate;
        L_stokes.cation  = CatIterate;
        L_stokes.anion   = AnIterate;
        L_stokes.phi     = esIterate;
        assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
        bc_stokes_2.apply(b_stokes);
        //bc_stokes_sides.apply(b_stokes); bc_stokes_topbottom.apply(b_stokes);

        L.uu     = VelocIterate;
        L.CatCat = CatIterate;
        L.AnAn   = AnIterate;
        L.EsEs   = esIterate;
        assemble(b,L); bc_1.apply(b); bc_2.apply(b);
        //bc_membrane_0.apply(b); bc_membrane_1.apply(b); bc_membrane_2.apply(b); bc_membrane_3.apply(b);
        //bc_protein_0.apply(b); bc_protein_1.apply(b); bc_protein_2.apply(b); bc_protein_3.apply(b);
        
        // convert right hand side and measure relative residual
        printf(" Update the nonlinear residual \n");
        // Stokes
        dvector newb_stokes_fasp;
        newb_stokes_fasp.row = b_stokes.size();
        newb_stokes_fasp.val = (double*)b_stokes.data();
        normb_stokes_fasp = fasp_blas_array_norm2(newb_stokes_fasp.row,newb_stokes_fasp.val);

        // PNP
        dvector newb_fasp;
        newb_fasp.row = b.size();
        newb_fasp.val = (double*)b.data();
        normb_fasp = fasp_blas_array_norm2(newb_fasp.row,newb_fasp.val);
        
        // Combined
        prev_relR = relR_fasp;
        relR_fasp = sqrt(normb_stokes_fasp*normb_stokes_fasp + normb_fasp*normb_fasp) / initial_residual;
        //relR_fasp = normb_fasp/b0_fasp ;



        printf(" After %d iteration(s) the relative residual is %e \n", it, relR_fasp);
        fflush(stdout);

        // Current

        /*
        printf(" Compute current in channel \n"); fflush(stdout);
        current_flux::Functional Current(mesh);
        double current;
        Constant n_vector(1.0,0.0,0.0);
        Current.n_vec = n_vector;
        Current.dS  = adapted_surfaces;
        Current.Dna = Dna; Current.qna = qna;
        Current.Dk  = Dk;  Current.qk  = qk;
        Current.Dcl = Dcl; Current.qcl = qcl;
        Current.Dca = Dca; Current.qca = qca;
        Current.Na  = CatIterate;
        Current.K   = AnIterate;
        Current.Cl  = ClIterate;
        Current.Ca  = CaIterate;
        Current.Es  = esIterate;
        current  = assemble(Current);
        current *= 1.0e-12*D_ref*p_ref*n_avo*e_chrg/L_ref;
        printf("   Current is computed as %e pA \n\n", current); fflush(stdout);
        */
        
        
        //************************
        //  Measure dissipation
        //************************
        
        // Convert from log to density
        double * CatConcval = CatConc.vector()->data();
        double * CatItval = CatIterate.vector()->data();
        for(std::size_t i=0; i<CatIterate.vector()->size(); ++i) {
            CatConcval[i] = exp(CatItval[i]);
        }
        double * AnConcval = AnConc.vector()->data();
        double * AnItval = AnIterate.vector()->data();
        for(std::size_t i=0; i<AnIterate.vector()->size(); ++i) {
            AnConcval[i] = exp(AnItval[i]);
        }        
        
        // Save solution in VTK format
        /*printf(" Write solution to file \n"); fflush(stdout);
         Nafile  << CatConc;
         Kfile   << AnConc;
         Cafile  << CaConc;
         Clfile  << ClConc;
         phifile << esIterate;
         
         
         Diss.Na = CatConc;
         Diss.K  = AnConc;
         Diss.Ca = CaConc;
         Diss.Cl = ClConc;
         Diss.phi = esIterate;
         diss     = assemble(Diss);
         printf(" The dissipation is %e \n\n", diss);
         fflush(stdout);
         */
        
        
        
  
        
        
        //**************************
        //  Check for convergence
        //**************************
        
        if ( relR_fasp < tol ) {       // Converged
            printf("\nConvergence: The relative residual is below the desired tolerance. \n\n");
            fflush(stdout);
            done = true;
            converged = true;
            
            //	Stop Timer
            gettimeofday(&tim, NULL) ;
            runtime -= tim.tv_sec+(tim.tv_usec/1000000.0) ;
            runtime *= -1. ;
            printf(" Runtime:  %e\n\n", runtime);
            
            
            
            // Save solution in VTK format
            printf(" Write solution to file \n"); fflush(stdout);
            Velocityfile << VelocIterate;
            Pressurefile << PressIterate;

            Catfile << CatConc;
            Anfile  << AnConc;
            phifile << esIterate;

            
            
            //  Write simulation summary to file
            /*
             printf(" Output for main-log-steady.cpp \n");fflush(stdout);
             FILE *fout;
             strcat(paramRegime,"-log-main-steady");
             fout = fopen(paramRegime, "w");
             fprintf(fout,"Output for main-log-steady.cpp \n\n");
             fprintf(fout," Mesh: ( %d, %d, %d ) \n", Nx,Ny,Nz);
             fprintf(fout," eps = %e \n", epsPVal);
             fprintf(fout," Dp  = %e \n", DpVal);
             fprintf(fout," Dn  = %e \n\n", DnVal);
             if (it>maxit)
             fprintf(fout," Status: No convergence \n");
             else
             fprintf(fout," Status: Convergence \n");
             fprintf(fout," H1-error:    %e \n", h1norm_error);
             fprintf(fout," Nonlin Res:  %e \n", normb_fasp);
             fprintf(fout," Rel Res:     %e \n", relR_fasp);
             fprintf(fout," Dissipation: %e \n", diss);
             fprintf(fout," Newton Iter: %d \n", it);
             fprintf(fout," Runtime:     %e\n\n", runtime);
             fclose(fout);
             */
            
            
        } else if ( it>=maxit ) {                    // Too many iterations
            printf("\nDivergence: The relative residual has not converged after %d iterations. \n\n",maxit+1);
            fflush(stdout);
            done=true;
        }  else if ( relR_fasp >= prev_relR ) {     // Increasing residual
            printf(" *** The relative residual has grown in this iteration *** \n\n");fflush(stdout);
        }

    
    }
    
    
    printf(" -------------------------------------\n"); fflush(stdout);
    printf(" End of Newton Iteration \n"); fflush(stdout);
    printf(" -------------------------------------\n\n"); fflush(stdout);
    
        
        
        
        
    
    //********************
    //  Adaptivity pass
    //********************
    
    printf(" Update mesh \n\n"); fflush(stdout);
    mesh0     = mesh;
    std::shared_ptr<const Mesh> adapted_mesh( new const Mesh(mesh0) );
    adaptFN   = adapt(esIterate, adapted_mesh);
    initVeloc = adapt(VelocIterate, adapted_mesh);
    initPress = adapt(PressIterate, adapted_mesh);
    initCat   = adapt(CatIterate, adapted_mesh);
    initAn    = adapt(AnIterate,  adapted_mesh);
    initPHI   = adapt(esIterate, adapted_mesh);
    
    }
    
    
    printf(" Failed to converge below desired relative residual!!! \n\n"); fflush(stdout);
    
    printf(" Exiting... \n\n"); fflush(stdout);
   
    
    return 0;
}