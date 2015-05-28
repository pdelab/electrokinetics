// Copyright (C) 2014 CCMA@PSU Maximilian Metti, Xiaozhe Hu


//*************************************************
//
//    Still need to improve the Convergence
//    criterion; the current implementation OF
//    relative residual will lead to oversolving
//
//**************************************************

#include <iostream>
#include <fstream>
#include <dolfin.h>
#include <sys/time.h>
#include <string.h>
#include "./include/dg_stokes.h"
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
        if ( fabs(x[0]) > 0.5 ) {
            values[bc_direction]  = outflow*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
            values[bc_direction] -=  inflow*(x[bc_direction]-bc_distance/2.0)/(bc_distance);
        }
    }
private:
    double outflow, inflow, bc_distance;
    int bc_direction;
};


//  Body force on Fluid
class BodyForce : public Expression
{
public:
    BodyForce(double out_flow, double in_flow, double bc_dist, int bc_dir): Expression(3),outflow(out_flow),inflow(in_flow),bc_distance(bc_dist),bc_direction(bc_dir) {}
    void eval(Array<double>& values, const Array<double>& x) const
    {
        values[0] =  0.0;
        values[1] =  0.0;
        values[2] =  0.0;
        //values[bc_direction]  = outflow*(x[bc_direction]+bc_distance/2.0)/(bc_distance);
        //values[bc_direction] -=  inflow*(x[bc_direction]-bc_distance/2.0)/(bc_distance) + .01*(x[bc_direction]-bc_distance/2.0)*(x[bc_direction]+bc_distance/2.0);
    }
private:
    double outflow, inflow, bc_distance;
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
        return on_boundary && (x[0] < -Lx+DOLFIN_EPS or x[0] > Lx-DOLFIN_EPS);
    }
};


// Sub domain for Dirichlet boundary condition
class ChannelWalls : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        bool sideWalls = (x[1] < -Ly+DOLFIN_EPS) or (x[1] > Ly-DOLFIN_EPS);
        bool bottom    = (x[2] < -Lz+DOLFIN_EPS);
        return on_boundary && (sideWalls or bottom);
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
    double mu; // relaxation parameter (damping factor)
    double adaptTol;
    
    // Boundary conditions
    int    bc_direction;
    double bc_distance;

    // PDE coefficients
    double viscosity;
    double penalty;
    
    char filenm[] = "./params/stokes-exp-names.dat";
    
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
    
    File meshfile("./output/stokes_adapt/mesh.pvd");
    File  Velocityfile("./output/stokes_adapt/VelFinal.pvd");
    File  Pressurefile("./output/stokes_adapt/PressFinal.pvd");
    
    
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
    
    // Reference values
    double v_ref  = 1.0e+0;             // Viscosity (Pa s = kg / s m)
    double L_ref  = 1.0e-09;            // length scale (m)
    double t_ref  = 1.334e-5;           // time scale
    
    printf(" The spatial scale is %e meters \n", L_ref); fflush(stdout);
    printf(" The time scale is %e seconds \n\n", t_ref); fflush(stdout);
    
    
    // PDE coefficients
    viscosity  = viscosity / v_ref;   // dim'less viscosity
    
    printf(" Dimensionless parameters are given by: \n"); fflush(stdout);
    printf("    Fluidic Viscosity:          \t %15.10f \n\n", viscosity); fflush(stdout);
    fflush(stdout);
    
    
    // Boundary conditions
    double in_flow  = 1.0;
    double out_flow = 1.0;
    
    printf(" The boundary conditions are: \n"); fflush(stdout);
    printf("    Fluidic Rate of Inflow:        %15.10f m/s \n", -in_flow); fflush(stdout);
    printf("    Fluidic Rate of Outflow:       %15.10f m/s \n", out_flow); fflush(stdout);
    
    
    
    //************************
    //  Analytic Expressions
    //************************
    printf("\n Initializing analytic expressions\n"); fflush(stdout);

    // Velocity
    printf("   Interpolating in/out flow rates for fluid \n"); fflush(stdout);
      FluidVelocity Velocity(out_flow,in_flow,bc_distance,bc_direction);
      BodyForce     bodyforce(out_flow,in_flow,bc_distance,bc_direction);
        
    
    
    //***************
    //  Time March
    //***************
    
    
    
    
    
    
    
    //**************
    //  Adaptivity
    //**************

    
    // Copy mesh and initialize CG space
    Mesh mesh0(initMesh);
    dg_stokes::FunctionSpace VQ0(mesh0);
    uint totalRefines = 0;
    
    // Define initial guesses and residual
    printf("\n Define initial guesses\n"); fflush(stdout);
    Function initGuessNS(VQ0);
    Constant zero(0.0);
    Function initVeloc(initGuessNS[0]); initVeloc.interpolate(Velocity);
    Function initPress(initGuessNS[1]); initPress.interpolate(zero);

    
    
    // Initialize adaptivity marker
    Function adaptFn0(VQ0);
    Function adaptFN(adaptFn0[1]);
    adaptFN.interpolate(initPress);
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
        printf(" Recovering gradient of pressure... \n"); fflush(stdout);
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
    dg_stokes::FunctionSpace VQ(mesh);
    
    // Define variational forms
    printf(" Define Stokes variational forms \n\n"); fflush(stdout);
    dg_stokes::BilinearForm a_stokes(VQ, VQ);
    dg_stokes::LinearForm L_stokes(VQ);


        
        
    //*********************************************
    //  Mark subdomains and impose Dirichlet B.C.
    //*********************************************
        
    // Define subdomains and subdomains
    printf("Define subdomains \n"); fflush(stdout);
    
    // Define Dirichlet boundary conditions
    printf(" Define Dirichlet boundary condition \n\n"); fflush(stdout);
      DirBCval DirBC;
      DirNoFlux noflux(mesh);
      DirichletBoundary inletoutlet;
      ChannelWalls channelwalls;

      FacetFunction<std::size_t> channelSurfaces(mesh);
      channelSurfaces.set_all(1);
      channelwalls.mark(channelSurfaces,2);
      inletoutlet.mark(channelSurfaces,3);
      meshfile << channelSurfaces;


      // Stokes Dirichlet BCs
      SubSpace V(VQ,0);
      //DirichletBC bc_stokes_0(V,DirBC,inletoutlet);
      //DirichletBC bc_stokes_2(V,noflux,channelwalls);
      DirichletBC bc_stokes_0(V,DirBC,adapted_surfaces,1);
      DirichletBC bc_stokes_1(V,DirBC,adapted_surfaces,2);
      DirichletBC bc_stokes_2(V,noflux,adapted_surfaces,3);
      std::vector<const DirichletBC*> bc_stokes;
      bc_stokes.push_back(&bc_stokes_0);
      bc_stokes.push_back(&bc_stokes_1);
      bc_stokes.push_back(&bc_stokes_2);

      

        
    
    
    //***********************
    // Assign coefficients
    //***********************
    
    printf("Assign coefficients for variational forms \n\n"); fflush(stdout);

    // Stokes eqns
        Constant visc(viscosity);     a_stokes.mu    = visc;  L_stokes.mu    = visc;    // Fluidic viscosity
        Constant alpha(penalty);      a_stokes.alpha = alpha; L_stokes.alpha = alpha;   // DG penalty
        Function forces(VQ);      
        Function bforce(forces[0]);   bforce.interpolate(bodyforce);
        L_stokes.F = bforce;                                                            // Body force
    
        
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
    double prev_relR, normb_stokes_fasp;
    double relR_fasp = 1.0;
    bool   done = false;
    
    // Block GS updates
    Function solu_stokes(VQ);

    // Newton updates
    Function newton_stokes(VQ);
    Function newton_veloc(newton_stokes[0]);
    Function newton_press(newton_stokes[1]);
    
    // Linear system
    Matrix A_stokes;
    Vector b_stokes;
        

    //  NS indices
    std::vector<dolfin::la_index> gidx_u;
    std::vector<dolfin::la_index> gidx_press;
    const dolfin::la_index n02 = VQ.dofmap()->ownership_range().first;
    const dolfin::la_index n12 = VQ.dofmap()->ownership_range().second;
    //const dolfin::la_index num_dofs2 = n12 - n02;
    std::vector<std::size_t> component(1);
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




    
    // Estimate the initial residual
    printf(" Measure initial residual \n");
        L_stokes.uu     = VelocIterate;
        L_stokes.pp     = PressIterate;    
        assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes);
        bc_stokes_1.apply(b_stokes); bc_stokes_2.apply(b_stokes);

        dvector b_stokes_fasp;
        b_stokes_fasp.row = b_stokes.size();
        b_stokes_fasp.val = (double*)b_stokes.data();
        if (b0_stokes_fasp < 0.) {   // Initial residual
            b0_stokes_fasp = fasp_blas_array_norm2(b_stokes_fasp.row,b_stokes_fasp.val);
            printf(" The initial Stokes residual is %e \n\n", b0_stokes_fasp);
        }

        double initial_residual = b0_stokes_fasp;


    
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
        
        
        // Update newton step
        it++;
        printf("Newton iteration %d\n", it); fflush(stdout);
        printf(" Construct Jacobian matrix \n"); fflush(stdout);

        // Update Stokes coefficients
        //a_stokes.uu = VelocIterate;
        assemble(A_stokes,a_stokes); bc_stokes_0.apply(A_stokes);
        bc_stokes_1.apply(A_stokes); bc_stokes_2.apply(A_stokes);


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
            
            
            
            // Assign MP
            /*
             unsigned int mnz = boost::tuples::get<3>(MassPress.data());
             int mrow = MassPress.size(0);
             int mcol = MassPress.size(1);
             int* map = (int*)fasp_mem_calloc(row+1, sizeof(int));
             const size_t* map_tmp = boost::tuples::get<0>(MassPress.data());
             for (ii=0; ii<mrow+1; ii++) {
             map[ii] = (int)map_tmp[ii];
             }
             int* mai = (int*)fasp_mem_calloc(mnz, sizeof(int));
             const size_t* mai_tmp = boost::tuples::get<1>(MassPress.data());
             for (ii=0; ii<mnz; ii++) {
             mai[ii] = (int)mai_tmp[ii];
             }
             double* max = (double*)boost::tuples::get<2>(MassPress.data());
             
             MP.row = mrow;
             MP.col = mcol;
             MP.nnz = mnz;
             MP.IA  = map;
             MP.JA  = mai;
             MP.val = max;
             */

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






        //********************
        //   Stokes Solve
        //********************
                
        printf("\n Step 3: solve the Stokes' linearized system\n"); fflush(stdout);
        INT flag = 0;
        //flag = fasp_solver_bdcsr_krylov_navier_stokes_with_pressure_mass(&A2bcsr, &b2bcsr, &NSsoluvec, &itparam, &amgparam, &iluparam, &schparam, &MP);
        flag = fasp_solver_bdcsr_krylov_navier_stokes(&A_stokes_bcsr, &b_stokes_bcsr, &soluvec_stokes, &itparam, &amgparam, &iluparam, &schparam);
        if (flag<0) {
            printf("\n### WARNING: Solver failed! Exit status = %d.\n\n", flag); fflush(stdout);
        }
        else {
            printf("\nSolver finished successfully!\n\n"); fflush(stdout);
        }


        // Convert solution vector to FE solution
        printf(" Step 4: convert Stokes' solution to FEniCS structure\n"); fflush(stdout);

        // Convert Stokes solution
        double * NSsolval = solu_stokes.vector()->data();
        for (std::size_t i=0; i<nrow_u; ++i)
            NSsolval[gidx_u[i]] = soluvec_stokes.val[i];
        for (std::size_t i=0; i<nrow_press; ++i)
            NSsolval[gidx_press[i]]   = soluvec_stokes.val[nrow_u+i];

        // Add to Stokes nonlinear update
        newton_veloc.interpolate(solu_stokes[0]);
        newton_press.interpolate(solu_stokes[1]);


        




        
        printf(" Step 5: free memory\n"); fflush(stdout);
        // Free memory
        fasp_bdcsr_free(&A_stokes_bcsr);
        fasp_dvec_free(&b_stokes_bcsr);
        fasp_ivec_free(&u_idx);
        fasp_ivec_free(&press_idx);
        fasp_dvec_free(&soluvec_stokes);
        free(ap2); free(ai2);
        //free(map); free(mai);
        
        printf(" ---------------------------\n"); fflush(stdout);
        printf(" End of interface to FASP \n"); fflush(stdout);
        printf(" ---------------------------\n\n"); fflush(stdout);

        
        
        
        //***************************
        //      Update solution
        //***************************
        
        printf(" Update solution \n"); fflush(stdout);

        // Stokes Update
        //Function stokes_update(VQ);
        Function dVeloc(newton_veloc);  //   updateVEL.interpolate(solu_stokes[0]);
        Function dPress(newton_press);// updatePRESS.interpolate(solu_stokes[1]);
        
        //  Backtrack line search
        printf("   Use backtracking to guarantee decreasing residual \n"); fflush(stdout);
        // stokes
        Function update_stokes(VQ);
        Function updateVELOC(update_stokes[0]);
        Function updatePRESS(update_stokes[1]);

        double test_stokes;
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
            
            // Evaluate residual
            L_stokes.uu      = updateVELOC;
            L_stokes.pp      = updatePRESS;
            assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes); 
            bc_stokes_1.apply(b_stokes); bc_stokes_2.apply(b_stokes);

            
            // Compute relative residual for backtrack line search
            // Stokes
            dvector newb_stokes_faspBT;
            newb_stokes_faspBT.row = b_stokes.size();
            newb_stokes_faspBT.val = (double*)b_stokes.data();
            double normb_stokes_faspBT = fasp_blas_array_norm2(newb_stokes_faspBT.row,newb_stokes_faspBT.val);
            test_stokes  = normb_stokes_faspBT/b0_stokes_fasp ;

            testRelRes = test_stokes;
            printf("   Stokes' residual after update is %e \n", test_stokes);
            fflush(stdout);
            
            // check for reduced residual or NaN
            if ( (testRelRes>relR_fasp-DOLFIN_EPS) || (testRelRes!=testRelRes) ) {
                printf("   Reducing the update by a factor of %e \n", dampFactor); fflush(stdout);
                dampFactor            *= 0.5;

                *(dVeloc.vector())    *= dampFactor;
                *(dPress.vector())    *= dampFactor;

            } else {// Significant reduction
                reduced = true;
            }
            if (dampFactor<1.0e-7)   tooDamped = true;  // too much backtracking
            if (reduced || tooDamped) reducing = false; // end while
        }
        
        // If backtrack failed
        if ( testRelRes > relR_fasp-DOLFIN_EPS ){
            printf("   BAD UPDATE!!! \n"); fflush(stdout);
        }

        
        // Update solutions
        *(VelocIterate.vector()) = *(updateVELOC.vector());
        *(PressIterate.vector()) = *(updatePRESS.vector());


        // Update residual
        printf(" Update variational forms \n"); fflush(stdout);
        L_stokes.uu      = VelocIterate;
        L_stokes.pp      = PressIterate;
        assemble(b_stokes,L_stokes); bc_stokes_0.apply(b_stokes);
        bc_stokes_1.apply(b_stokes); bc_stokes_2.apply(b_stokes);

        
        // convert right hand side and measure relative residual
        printf(" Update the nonlinear residual \n");
        // Stokes
        dvector newb_stokes_fasp;
        newb_stokes_fasp.row = b_stokes.size();
        newb_stokes_fasp.val = (double*)b_stokes.data();
        normb_stokes_fasp = fasp_blas_array_norm2(newb_stokes_fasp.row,newb_stokes_fasp.val);

        prev_relR = relR_fasp;
        relR_fasp = normb_stokes_fasp / initial_residual;
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
    adaptFN   = adapt(PressIterate, adapted_mesh);
    initVeloc = adapt(VelocIterate, adapted_mesh);
    initPress = adapt(PressIterate, adapted_mesh);
    
    }
    
    
    printf(" Failed to converge below desired relative residual!!! \n\n"); fflush(stdout);
    
    printf(" Exiting... \n\n"); fflush(stdout);
   
    
    return 0;
}